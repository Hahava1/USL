# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict
import torch
from detectron2.layers import cat, ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import (
    select_foreground_proposals,
    StandardROIHeads,
)
from detectron2.structures import Instances, pairwise_iou
from fvcore.nn import smooth_l1_loss
from meshrcnn.modeling.roi_heads.mask_head import mask_rcnn_loss
from meshrcnn.modeling.roi_heads.mesh_head import (
    build_mesh_head,
    mesh_rcnn_inference,
    # mesh_rcnn_loss, # Não importamos a perda 3D original
)
from meshrcnn.modeling.roi_heads.voxel_head import (
    build_voxel_head,
    voxel_rcnn_inference,
    voxel_rcnn_loss, # Mantemos a perda voxel por enquanto (pode comentar se crashar)
)
from meshrcnn.modeling.roi_heads.layout_head import (
    build_layout_head,
    layout_rcnn_inference,
    # layout_rcnn_loss, # Não usamos a perda 3D original
)
from meshrcnn.utils import vis as vis_utils
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
# Removemos importações de renderização e perda 2D
from torch.nn import functional as F


@ROI_HEADS_REGISTRY.register()
class MeshRCNNROIHeads(StandardROIHeads):
    """
    Versão simplificada para demonstração:
    - Usa LayoutHead (z, rho).
    - Usa MeshHead com RoIAlign (original, sem RoIMap).
    - Remove perdas 3D (Chamfer, Normals, Edge) e perda USL 2D.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_layout_head(cfg, input_shape)
        self._init_voxel_head(cfg, input_shape)
        # [MODIFICADO] Reverte _init_mesh_head para usar pooler
        self._init_mesh_head_original_pooler(cfg, input_shape) # Usa função revertida
        self._vis = cfg.MODEL.VIS_MINIBATCH; self._misc = {}; self._vis_dir = cfg.OUTPUT_DIR

    # --- (_init_layout_head, _init_voxel_head inalterados) ---
    def _init_layout_head(self, cfg, input_shape):
        self.layout_on = cfg.MODEL.LAYOUT_ON;
        if not self.layout_on: return
        layout_pooler_resolution = cfg.MODEL.ROI_LAYOUT_HEAD.POOLER_RESOLUTION; layout_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        layout_sampling_ratio = cfg.MODEL.ROI_LAYOUT_HEAD.POOLER_SAMPLING_RATIO; layout_pooler_type = cfg.MODEL.ROI_LAYOUT_HEAD.POOLER_TYPE
        self.layout_loss_weight = cfg.MODEL.ROI_LAYOUT_HEAD.Z_REG_WEIGHT; self.layout_smooth_l1_beta = cfg.MODEL.ROI_LAYOUT_HEAD.SMOOTH_L1_BETA
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.layout_pooler = ROIPooler(output_size=layout_pooler_resolution, scales=layout_pooler_scales, sampling_ratio=layout_sampling_ratio, pooler_type=layout_pooler_type)
        shape = ShapeSpec(channels=in_channels, width=layout_pooler_resolution, height=layout_pooler_resolution)
        self.layout_head = build_layout_head(cfg, shape)
    def _init_voxel_head(self, cfg, input_shape):
        self.voxel_on = cfg.MODEL.VOXEL_ON;
        if not self.voxel_on: return
        voxel_pooler_resolution = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_RESOLUTION; voxel_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        voxel_sampling_ratio = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_SAMPLING_RATIO; voxel_pooler_type = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_TYPE
        self.voxel_loss_weight = cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT; self.cls_agnostic_voxel = cfg.MODEL.ROI_VOXEL_HEAD.CLS_AGNOSTIC_VOXEL
        self.cubify_thresh = cfg.MODEL.ROI_VOXEL_HEAD.CUBIFY_THRESH; in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.voxel_pooler = ROIPooler(output_size=voxel_pooler_resolution, scales=voxel_pooler_scales, sampling_ratio=voxel_sampling_ratio, pooler_type=voxel_pooler_type)
        shape = ShapeSpec(channels=in_channels, width=voxel_pooler_resolution, height=voxel_pooler_resolution)
        self.voxel_head = build_voxel_head(cfg, shape)

    # [MODIFICADO] Função revertida para usar o pooler original
    def _init_mesh_head_original_pooler(self, cfg, input_shape):
        self.mesh_on        = cfg.MODEL.MESH_ON
        if not self.mesh_on: return
        # Readiciona config do pooler
        mesh_pooler_resolution  = cfg.MODEL.ROI_MESH_HEAD.POOLER_RESOLUTION
        mesh_pooler_scales      = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        mesh_sampling_ratio     = cfg.MODEL.ROI_MESH_HEAD.POOLER_SAMPLING_RATIO
        mesh_pooler_type        = cfg.MODEL.ROI_MESH_HEAD.POOLER_TYPE

        self.chamfer_loss_weight = cfg.MODEL.ROI_MESH_HEAD.CHAMFER_LOSS_WEIGHT
        self.normals_loss_weight = cfg.MODEL.ROI_MESH_HEAD.NORMALS_LOSS_WEIGHT
        self.edge_loss_weight = cfg.MODEL.ROI_MESH_HEAD.EDGE_LOSS_WEIGHT
        self.gt_num_samples = cfg.MODEL.ROI_MESH_HEAD.GT_NUM_SAMPLES
        self.pred_num_samples = cfg.MODEL.ROI_MESH_HEAD.PRED_NUM_SAMPLES
        self.gt_coord_thresh = cfg.MODEL.ROI_MESH_HEAD.GT_COORD_THRESH
        self.ico_sphere_level = cfg.MODEL.ROI_MESH_HEAD.ICO_SPHERE_LEVEL

        # Readiciona o mesh_pooler
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.mesh_pooler = ROIPooler(
            output_size=mesh_pooler_resolution,
            scales=mesh_pooler_scales,
            sampling_ratio=mesh_sampling_ratio,
            pooler_type=mesh_pooler_type,
        )
        # Build mesh_head com shape do pooler
        self.mesh_head = build_mesh_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=mesh_pooler_resolution,
                width=mesh_pooler_resolution,
            ),
        )

    def forward(self, images, features, proposals, targets=None):
        if self._vis: self._misc["images"] = images
        if self.training: original_targets = targets; proposals = self.label_and_sample_proposals(proposals, targets); targets = original_targets
        if self._vis: self._misc["proposals"] = proposals
        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_layout(features, proposals)) # Mantém Layout
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_shape(features, proposals, targets, images)) # Chama shape simplificado
            # if self._vis: vis_utils.visualize_minibatch(...)
            return [], losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        assert not self.training; assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_layout(features, instances) # Mantém Layout
        instances = self._forward_mask(features, instances)
        instances = self._forward_shape(features, instances, None, None) # Chama shape simplificado
        return instances

    # --- (_forward_layout, _forward_mask inalterados) ---
    def _forward_layout(self, features, instances):
        # (Código inalterado da versão anterior)
        if not self.layout_on: return {} if self.training else instances
        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            proposal_boxes_tensors = [box.tensor for box in proposal_boxes if box.tensor.numel() > 0]
            if not proposal_boxes_tensors or cat(proposal_boxes_tensors, dim=0).numel() == 0: return {}
            layout_features = self.layout_pooler(features_list, proposal_boxes); layout_preds = self.layout_head(layout_features)
            pred_z, pred_rho = layout_preds
            num_proposals_per_image = [len(inst) for inst in instances]
            pred_z_list = pred_z.split(num_proposals_per_image, dim=0); pred_rho_list = pred_rho.split(num_proposals_per_image, dim=0)
            for instances_per_image, z, rho in zip(instances, pred_z_list, pred_rho_list): instances_per_image.pred_z = z; instances_per_image.pred_rho = rho
            return {}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_boxes_tensors = [box.tensor for box in pred_boxes if box.tensor.numel() > 0]
            if not pred_boxes_tensors or cat(pred_boxes_tensors, dim=0).numel() == 0:
                for inst in instances: inst.pred_z = torch.zeros((0, 1), device=features_list[0].device); inst.pred_rho = torch.zeros((0, 1), device=features_list[0].device)
                return instances
            layout_features = self.layout_pooler(features_list, pred_boxes); layout_preds = self.layout_head(layout_features)
            layout_rcnn_inference(layout_preds, instances); return instances
    def _forward_mask(self, features, instances):
         # (Código inalterado da versão anterior)
        if not self.mask_on: return {} if self.training else instances
        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes_tensors = [p.proposal_boxes.tensor for p in proposals if p.proposal_boxes.tensor.numel() > 0]
            if not proposal_boxes_tensors or cat(proposal_boxes_tensors, dim=0).numel() == 0: return {"loss_mask": torch.tensor(0.0, device=features_list[0].device)}
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features_list, proposal_boxes)
            mask_logits = self.mask_head.layers(mask_features); loss_mask, target_masks = mask_rcnn_loss(mask_logits, proposals)
            if self._vis: self._misc["target_masks"] = target_masks; self._misc["fg_proposals"] = proposals
            return {"loss_mask": loss_mask}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_boxes_tensors = [box.tensor for box in pred_boxes if box.tensor.numel() > 0]
            if not pred_boxes_tensors or cat(pred_boxes_tensors, dim=0).numel() == 0:
                mask_shape = self.mask_head.output_shape if hasattr(self.mask_head, 'output_shape') else (1, 256, 14, 14)
                for inst in instances: inst.pred_masks = torch.zeros((0,) + mask_shape[1:], device=features_list[0].device)
                return instances
            mask_features = self.mask_pooler(features_list, pred_boxes); return self.mask_head(mask_features, instances)

    # [MODIFICADO] _forward_shape revertido para usar mesh_pooler
    # e remove cálculo de perdas 3D e USL 2D
    def _forward_shape(self, features, instances, targets=None, images=None):
        if not self.voxel_on and not self.mesh_on:
            return {} if self.training else instances

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # Seleciona propostas foreground (necessário para Voxel loss se ativa)
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            losses = {}
            init_mesh = Meshes(verts=[], faces=[]) # Inicializa mesh vazia

            if self.voxel_on:
                 # Calcula Voxel loss se voxel_on=True
                 voxel_features = self.voxel_pooler(features_list, proposal_boxes)
                 voxel_logits = self.voxel_head(voxel_features)
                 loss_voxel, target_voxels = voxel_rcnn_loss(
                     voxel_logits, proposals, loss_weight=self.voxel_loss_weight
                 )
                 losses.update({"loss_voxel": loss_voxel}) # Mantém Voxel Loss por agora
                 if self._vis: self._misc["target_voxels"] = target_voxels
                 # Cria malha inicial a partir dos voxels previstos (necessário se mesh_on=True)
                 if self.mesh_on:
                      if self.cls_agnostic_voxel:
                          with torch.no_grad():
                              vox_in = voxel_logits.sigmoid().squeeze(1)
                              init_mesh = cubify(vox_in, self.cubify_thresh)
                      else: raise ValueError("...")

            if self.mesh_on:
                # Usa o mesh_pooler (RoIAlign)
                mesh_features = self.mesh_pooler(features_list, proposal_boxes)

                if not self.voxel_on: # Cria ico_sphere se voxel estiver desligado
                     if mesh_features.shape[0] > 0:
                         init_mesh = ico_sphere(self.ico_sphere_level, mesh_features.device)
                         init_mesh = init_mesh.extend(mesh_features.shape[0])
                     else: init_mesh = Meshes(verts=[], faces=[])

                # Executa a mesh_head (agora espera features do pooler)
                pred_meshes_list = self.mesh_head(mesh_features, init_mesh) # Passa mesh_features

                # [MODIFICADO] NÃO CALCULA NENHUMA PERDA DE MALHA (nem 3D nem 2D)
                # Apenas guarda a previsão para possível visualização/inferência
                pred_meshes_final = pred_meshes_list[-1]
                num_proposals_per_image_fg = [len(p) for p in proposals]
                pred_meshes_split = pred_meshes_final.split(num_proposals_per_image_fg)
                for proposals_per_image, mesh in zip(proposals, pred_meshes_split):
                    proposals_per_image.pred_meshes = mesh # Guarda para inferência

                # Retorna apenas a voxel_loss (se ativa), ou vazio
                dummy_param = next(self.mesh_head.parameters())
                losses.update({ # Adiciona perdas zero para evitar erros no backward
                    "loss_usl_xent": dummy_param.sum() * 0.0,
                    "loss_usl_dist": dummy_param.sum() * 0.0
                })

            return losses # Retorna apenas perdas existentes (box, mask, talvez voxel)
        else:
            # Lógica de Inferência (revertida para usar pooler)
            pred_boxes = [x.pred_boxes for x in instances]
            init_mesh = Meshes(verts=[], faces=[])

            if self.voxel_on:
                pred_boxes_tensors = [box.tensor for box in pred_boxes if box.tensor.numel() > 0]
                if pred_boxes_tensors and cat(pred_boxes_tensors, dim=0).numel() > 0:
                    voxel_features = self.voxel_pooler(features_list, pred_boxes)
                    voxel_logits = self.voxel_head(voxel_features)
                    voxel_rcnn_inference(voxel_logits, instances)
                    if self.cls_agnostic_voxel:
                        with torch.no_grad(): vox_in = voxel_logits.sigmoid().squeeze(1); init_mesh = cubify(vox_in, self.cubify_thresh)
                    else: raise ValueError("...")
                else: init_mesh = Meshes(verts=[], faces=[])

            if self.mesh_on:
                # Usa mesh_pooler
                pred_boxes_tensors = [box.tensor for box in pred_boxes if box.tensor.numel() > 0]
                if pred_boxes_tensors and cat(pred_boxes_tensors, dim=0).numel() > 0:
                    mesh_features = self.mesh_pooler(features_list, pred_boxes)
                    if not self.voxel_on:
                        if mesh_features.shape[0] > 0: init_mesh = ico_sphere(self.ico_sphere_level, mesh_features.device); init_mesh = init_mesh.extend(mesh_features.shape[0])
                        else: init_mesh = Meshes(verts=[], faces=[])
                    # Executa mesh_head com features do pooler
                    pred_meshes = self.mesh_head(mesh_features, init_mesh)
                    mesh_rcnn_inference(pred_meshes[-1], instances)
                else: # Se não houver caixas, anexa None
                    for inst in instances: inst.pred_meshes = None

            else: # Se mesh_on=False mas voxel_on=True
                 if self.voxel_on: mesh_rcnn_inference(init_mesh, instances)

            return instances
