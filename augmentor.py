from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CropAndPad(
        percent=(-0.25, 0.25),
        pad_mode="constant",
        pad_cval=0
    ),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
        order=[0],
        mode='constant',
        cval=0
    )
])
seq_det = seq.to_deterministic()

seq_img = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

], random_order=True)
