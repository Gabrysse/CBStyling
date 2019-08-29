import argparse
import utils as ut


def main(img_fname, style_id, class_id):
    # Create style and segmentation model
    seg_model = ut.create_seg_model()
    style_model = ut.create_style_model(style_id)

    # ======================================
    # Seg part
    image = ut.load_image(img_fname)
    semseg = ut.get_semseg_image(seg_model, image)
    # Get image with mask showing class_id 11, which is people by default
    print(class_id)
    fg_image = ut.get_masked_image(seg_model, image, class_id=class_id, bg=0)
    # Get image with mask showing everything except class_id 13
    bg_image = ut.get_masked_image(seg_model, image, class_id=class_id, bg=1)

    # ======================================
    # Style part
    image = ut.load_image_style(img_fname, scale=1.0)
    image_style1 = ut.get_styled_image(style_model, image)
    # Apply local style to fg
    fg_styled = image_style1 * (fg_image != 0)
    # Apply local style to bg
    bg_styled = image_style1 * (bg_image != 0)

    # ======================================
    # Save part
    ut.save_image('class_id_%d_original.png' %class_id, ut.f2l(image[0].cpu().numpy()))
    ut.save_image('class_id_%d_styled_fg.png' %class_id, fg_styled + bg_image)
    ut.save_image('class_id_%d_fg.png' %class_id, image_style1)
    ut.save_image('class_id_%d_semseg.png' %class_id, ut.f2l(ut.VOCColorize()(semseg)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--img-path", type=str, default="", help="path of image to style")
    parser.add_argument("-s", "--style", type=int, default=0, help="Choose a style 0-3")
    parser.add_argument("-c", "--class_id", type=int, default=11, help="Choose a class_id 1-20")
    args = parser.parse_args()

    main(args.img_fname, args.style, args.class_id)