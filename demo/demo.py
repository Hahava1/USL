#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import logging
import multiprocessing as mp
import os

import cv2

# required so that .register() calls are executed in module scope
import meshrcnn.data  # noqa
import meshrcnn.modeling  # noqa
import meshrcnn.utils  # noqa
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from meshrcnn.config import get_meshrcnn_cfg_defaults
from meshrcnn.evaluation import transform_meshes_to_camera_coord_system
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
# [ADICIONADO] Importa cat
from detectron2.layers import cat


logger = logging.getLogger("demo")


class VisualizationDemo:
    def __init__(self, cfg, vis_highest_scoring=True, output_dir="./vis"):
        """
        Args:
            cfg (CfgNode):
            vis_highest_scoring (bool): If set to True visualizes only
                                        the highest scoring prediction
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.colors = self.metadata.thing_colors
        self.cat_names = self.metadata.thing_classes

        self.cpu_device = torch.device("cpu")
        self.vis_highest_scoring = vis_highest_scoring
        self.predictor = DefaultPredictor(cfg)

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def run_on_image(self, image, focal_length=10.0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            focal_length (float): the focal_length of the image

        Returns:
            predictions (dict): the output of the model.
        """
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]

        # camera matrix
        imsize = [image.shape[0], image.shape[1]]
        # focal <- focal * image_width / 32
        focal_length = image.shape[1] / 32 * focal_length
        K = [focal_length, image.shape[1] / 2, image.shape[0] / 2]

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            scores = instances.scores
            boxes = instances.pred_boxes
            labels = instances.pred_classes
            masks = instances.pred_masks

            # Verifica se há previsões de malha antes de tentar aceder
            if not instances.has("pred_meshes") or len(instances.pred_meshes) == 0:
                 print("Nenhuma malha prevista para esta imagem.")
                 return predictions # Retorna sem tentar processar malhas

            # Cria o objeto Meshes apenas se houver previsões
            meshes = Meshes(
                verts=[mesh[0] for mesh in instances.pred_meshes],
                faces=[mesh[1] for mesh in instances.pred_meshes],
            )

            # [MODIFICADO] Calcula zranges a partir de pred_z e pred_rho
            if instances.has("pred_z") and instances.has("pred_rho"):
                pred_z_multiclass = instances.pred_z # (N, C) ou (N, 1)
                pred_rho_multiclass = instances.pred_rho # (N, C) ou (N, 1)
                pred_classes = instances.pred_classes # (N,)

                # Seleciona a previsão de z/rho correta com base na classe
                num_preds = pred_z_multiclass.shape[0]
                if pred_z_multiclass.shape[1] > 1: # Se não for agnóstico de classe
                    indices = torch.arange(num_preds, device=self.cpu_device)
                    pred_z = pred_z_multiclass[indices, pred_classes] # -> (N,)
                    pred_rho = pred_rho_multiclass[indices, pred_classes] # -> (N,)
                else: # Se for agnóstico (shape (N, 1))
                    pred_z = pred_z_multiclass.squeeze(-1) # -> (N,)
                    pred_rho = pred_rho_multiclass.squeeze(-1) # -> (N,)

                # Converte z (0, 1) e rho (0, 1) para o espaço do mundo
                # (Estes valores são hiperparâmetros, devem coincidir com project_scene)
                z0, z1 = 1.0, 10.0 # Ex: Profundidade min/max de 1m a 10m
                rho0, rho1 = 0.1, 5.0 # Ex: Extensão min/max de 0.1m a 5m
                z_world = z0 + pred_z * (z1 - z0)
                rho_world = rho0 + pred_rho * (rho1 - rho0)

                # Aproxima zranges a partir de z_world e rho_world
                z_min = z_world - rho_world / 2.0
                z_max = z_world + rho_world / 2.0
                zranges = torch.stack([z_min, z_max], dim=1) # (N, 2)

                # Garante que zranges não seja negativo (pode acontecer se rho for grande)
                zranges = torch.clamp(zranges, min=1e-3)

            else:
                 print("AVISO: 'pred_z' ou 'pred_rho' não encontrados. Usando zranges padrão.")
                 # Fallback se pred_z/pred_rho não existirem (não deveria acontecer)
                 zranges = torch.tensor([[1.0, 10.0]]).expand(len(meshes), 2).to(self.cpu_device)


            # [REMOVIDO] Bloco original que usava pred_dz
            # pred_dz = instances.pred_dz[:, 0] * (
            #     (boxes.tensor[:, 3] - boxes.tensor[:, 1]) / focal_length
            # )
            # tc = pred_dz.abs().max() + 1.0 # tc não é mais necessário
            # zranges = torch.stack(
            #     [
            #         torch.stack(
            #             [
            #                 tc - tc * pred_dz[i] / 2.0 / focal_length,
            #                 tc + tc * pred_dz[i] / 2.0 / focal_length,
            #             ]
            #         )
            #         for i in range(len(meshes))
            #     ],
            #     dim=0,
            # )

            Ks = torch.tensor(K).to(self.cpu_device).view(1, 3).expand(len(meshes), 3)

            # A função de transformação deve funcionar com os novos zranges
            meshes = transform_meshes_to_camera_coord_system(
                meshes, boxes.tensor, zranges, Ks, imsize
            )

            if self.vis_highest_scoring:
                # Lida com caso de não haver scores (raro, mas defensivo)
                if len(scores) > 0:
                     det_ids = [scores.argmax().item()]
                else:
                     det_ids = []
            else:
                det_ids = range(len(scores))

            for det_id in det_ids:
                 # Verifica se det_id é válido para todos os arrays
                 if det_id < len(boxes.tensor) and \
                    det_id < len(labels) and \
                    det_id < len(scores) and \
                    det_id < len(masks) and \
                    det_id < len(meshes):
                     self.visualize_prediction(
                         det_id,
                         image,
                         boxes.tensor[det_id],
                         labels[det_id],
                         scores[det_id],
                         masks[det_id],
                         meshes[det_id],
                     )

        return predictions

    # --- (Função visualize_prediction inalterada) ---
    def visualize_prediction(
        self, det_id, image, box, label, score, mask, mesh, alpha=0.6, dpi=200
    ):
        mask_color = np.array(self.colors[label], dtype=np.float32)
        cat_name = self.cat_names[label]
        thickness = max([int(np.ceil(0.001 * image.shape[0])), 1])
        box_color = (0, 255, 0)  # '#00ff00', green
        text_color = (218, 227, 218)  # gray

        composite = image.copy().astype(np.float32)

        # overlay mask
        idx = mask.nonzero(as_tuple=True) # Usa as_tuple=True para PyTorch > 1.2
        composite[idx[0], idx[1], :] *= 1.0 - alpha
        composite[idx[0], idx[1], :] += alpha * mask_color

        # overlay box
        (x0, y0, x1, y1) = (int(x.item() + 0.5) for x in box) # Usa .item()
        composite = cv2.rectangle(composite, (x0, y0), (x1, y1), color=box_color, thickness=thickness)
        composite = composite.astype(np.uint8)

        # overlay text
        font_scale = max(0.0005 * image.shape[0], 0.4) # Ajusta escala mínima
        font_thickness = thickness
        font = cv2.FONT_HERSHEY_SIMPLEX # Muda fonte para uma mais comum
        text = "%s %.3f" % (cat_name, score)
        ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, font_thickness)
        if x0 + text_w > composite.shape[1]: x0 = composite.shape[1] - text_w
        if y0 - int(1.2 * text_h) < 0: y0 = int(1.2 * text_h)
        back_topleft = x0, y0 - int(1.3 * text_h); back_bottomright = x0 + text_w, y0
        cv2.rectangle(composite, back_topleft, back_bottomright, box_color, -1)
        text_bottomleft = x0, y0 - int(0.2 * text_h)
        cv2.putText(composite, text, text_bottomleft, font, font_scale, text_color, thickness=font_thickness, lineType=cv2.LINE_AA)

        save_file = os.path.join(self.output_dir, "%d_mask_%s_%.3f.png" % (det_id, cat_name, score))
        cv2.imwrite(save_file, composite[:, :, ::-1])

        save_file = os.path.join(self.output_dir, "%d_mesh_%s_%.3f.obj" % (det_id, cat_name, score))
        # Verifica se mesh não está vazio
        if not mesh.isempty():
             verts, faces = mesh.get_mesh_verts_faces(0)
             save_obj(save_file, verts, faces)


# --- (Funções setup_cfg, get_parser, main inalteradas) ---
def setup_cfg(args):
    cfg = get_cfg()
    get_meshrcnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="MeshRCNN Demo")
    parser.add_argument("--config-file", default="configs/pix3d/meshrcnn_R50_FPN.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--input", required=True, help="A path to an input image") # Torna input obrigatório
    parser.add_argument("--output", required=True, help="A directory to save output visualizations") # Torna output obrigatório
    parser.add_argument("--focal-length", type=float, default=20.0, help="Focal length for the image")
    parser.add_argument("--onlyhighest", action="store_true", help="will return only the highest scoring detection")
    parser.add_argument("opts", help="Modify model config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser

def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger(name="demo")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Verifica se input é um ficheiro
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    im_name = os.path.splitext(os.path.basename(args.input))[0]

    demo = VisualizationDemo(cfg, vis_highest_scoring=args.onlyhighest, output_dir=os.path.join(args.output, im_name))

    img = read_image(args.input, format="BGR")
    predictions = demo.run_on_image(img, focal_length=args.focal_length)
    logger.info("Predictions saved in %s" % (os.path.join(args.output, im_name)))


if __name__ == "__main__":
    main()  # pragma: no cover
