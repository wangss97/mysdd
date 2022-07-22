from model.deeplab.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.wide_resnet50_2(output_stride, BatchNorm, False)
    elif backbone == 'xception':    # x
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'drn':         #  x
        return drn.drn_d_54(BatchNorm, pretrained=False)
    elif backbone == 'mobilenet':       #   x
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
