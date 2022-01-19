# ------------------------------------------------------------------------
# DT-MIL
# Copyright (c) 2021 Tencent. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


from .deformable_detr_wsi_feat import build as build_wsi_feat
from .transformer import build as build_pure_transformer

__all__ = ['build_pure_transformer', 'build_wsi_feat']


def build_wsi_feat_model(args):
    return build_wsi_feat(args)
