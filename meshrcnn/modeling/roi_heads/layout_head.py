# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.layers import cat, ShapeSpec
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

# Modificado de Z_HEAD para LAYOUT_HEAD
ROI_LAYOUT_HEAD_REGISTRY = Registry("ROI_LAYOUT_HEAD")


@ROI_LAYOUT_HEAD_REGISTRY.register()
class FastRCNNFCHead(nn.Module):
    """
    A head with several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_fc: the number of fc layers
            fc_dim: the dimension of the fc layers
        """
        super().__init__()

        # fmt: off
        # Modificado de ROI_Z_HEAD para ROI_LAYOUT_HEAD
        num_fc          = cfg.MODEL.ROI_LAYOUT_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_LAYOUT_HEAD.FC_DIM
        cls_agnostic    = cfg.MODEL.ROI_LAYOUT_HEAD.CLS_AGNOSTIC_Z_REG
        num_classes     = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # fmt: on

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            # Modificado de z_fc para layout_fc
            self.add_module("layout_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        num_layout_reg_classes = 1 if cls_agnostic else num_classes
        
        # Renomeado de z_pred para z_head
        self.z_head = nn.Linear(fc_dim, num_layout_reg_classes)
        # ADICIONADO: Nova cabeça para prever rho (extensão)
        self.rho_head = nn.Linear(fc_dim, num_layout_reg_classes)


        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        # Renomeado de z_pred para z_head e adicionado rho_head
        nn.init.normal_(self.z_head.weight, std=0.001)
        nn.init.constant_(self.z_head.bias, 0)
        nn.init.normal_(self.rho_head.weight, std=0.001)
        nn.init.constant_(self.rho_head.bias, 0)

    # MODIFICADO: O 'forward' agora retorna duas saídas (z, rho)
    # e aplica sigmoid, conforme o artigo USL.
    def forward(self, x):
        x = x.view(x.shape[0], np.prod(x.shape[1:]))
        for layer in self.fcs:
            x = F.relu(layer(x))
        
        # Nossas duas novas saídas
        pred_z = self.z_head(x)
        pred_rho = self.rho_head(x)
        
        # Aplicar sigmoid como no paper USL (saída entre 0 e 1)
        pred_z = torch.sigmoid(pred_z)
        pred_rho = torch.sigmoid(pred_rho)

        return pred_z, pred_rho

    @property
    def output_size(self):
        return self._output_size


# Renomeado de z_rcnn_loss para layout_rcnn_loss
# Esta função não será usada no treinamento do USL,
# mas a mantemos para não quebrar a estrutura do código.
def layout_rcnn_loss(z_pred, instances, src_boxes, loss_weight=1.0, smooth_l1_beta=0.0):
    """
    Compute the z_pred loss.

    Args:
        z_pred (Tensor): A tensor of shape (B, C) or (B, 1) for class-specific or class-agnostic,
            where B is the total number of foreground regions in all images, C is the number of foreground classes,
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_z = z_pred.size(1) == 1
    total_num = z_pred.size(0)

    gt_classes = []
    gt_dz = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_z:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_dz.append(instances_per_image.gt_dz)

    if len(gt_dz) == 0:
        return z_pred.sum() * 0

    gt_dz = cat(gt_dz, dim=0)
    assert gt_dz.numel() > 0
    src_heights = src_boxes[:, 3] - src_boxes[:, 1]
    dz = torch.log(gt_dz / src_heights)

    if cls_agnostic_z:
        z_pred = z_pred[:, 0]
    else:
        indices = torch.arange(total_num)
        gt_classes = cat(gt_classes, dim=0)
        z_pred = z_pred[indices, gt_classes]

    loss_z_reg = smooth_l1_loss(z_pred, dz, smooth_l1_beta, reduction="sum")
    loss_z_reg = loss_weight * loss_z_reg / gt_classes.numel()
    return loss_z_reg


# Renomeado de z_rcnn_inference para layout_rcnn_inference
# MODIFICADO: para lidar com as duas saídas (pred_z, pred_rho)
def layout_rcnn_inference(preds, pred_instances):
    # 'preds' agora é uma tupla (pred_z, pred_rho) vinda do 'forward'
    pred_z, pred_rho = preds
    
    cls_agnostic = pred_z.size(1) == 1

    if not cls_agnostic:
        num_preds = pred_z.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_preds, device=class_pred.device)
        
        pred_z = pred_z[indices, class_pred][:, None]
        pred_rho = pred_rho[indices, class_pred][:, None]

    # O USL não usa a transformação log/exp,
    # então removemos essas linhas. Os valores já estão (0, 1).

    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_z = pred_z.split(num_boxes_per_image, dim=0)
    pred_rho = pred_rho.split(num_boxes_per_image, dim=0)

    for z_reg, rho_reg, instances in zip(pred_z, pred_rho, pred_instances):
        # Salva os novos campos nas instâncias
        instances.pred_z = z_reg
        instances.pred_rho = rho_reg


# Renomeado de build_z_head para build_layout_head
def build_layout_head(cfg, input_shape):
    # Modificado de ROI_Z_HEAD para ROI_LAYOUT_HEAD
    name = cfg.MODEL.ROI_LAYOUT_HEAD.NAME
    return ROI_LAYOUT_HEAD_REGISTRY.get(name)(cfg, input_shape)
