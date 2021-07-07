import argparse
import utils as ut
from torchvision import transforms
from PIL import Image


def main(img_fname, out_fname, style_id, class_id):
    # Create style and segmentation model
    seg_model = ut.create_seg_model()
    style_model = ut.create_style_model(style_id)

    # ======================================
    # Seg part
    image = ut.load_image(img_fname)
    print(image.shape)

    semseg = ut.get_semseg_image(seg_model, image)
    print(semseg.shape)

    # Get image with mask showing class_id 13, which is cars by default
    print("Stylizing class %d..." % class_id)
    fg_image = ut.get_masked_image(seg_model, image, category=class_id, bg=0)
    ut.save_image("images/fg_image.png", fg_image)

    # Get image with mask showing everything except class_id 13
    bg_image = ut.get_masked_image(seg_model, image, category=class_id, bg=1)
    ut.save_image("images/bg_image.png", bg_image)

    # ======================================
    # Style part
    # load_image_style carica l'immagine in input, la mette in un tensore e la
    # carica nella GPU
    # size = torch.Size([1, 3, 1024, 2048])
    image = ut.load_image_style(img_fname, scale=1.0)
    print(image.size())

    # Qui invece viene applicato lo stile all'immagine caricata prima
    image_style1 = ut.get_styled_image(style_model, image)

    # Apply local style to fg
    # Applica lo stile solo alla classe scelta 
    fg_styled = image_style1 * (fg_image != 0)
    ut.save_image("images/fg_styled.png", fg_styled)

    # Apply local style to bg
    # Applica lo stile a tutto il resto dell'immagine
    bg_styled = image_style1 * (bg_image != 0)
    ut.save_image("images/bg_styled.png", bg_styled)

    # ======================================
    # Save part
    # ut.save_image(out_fname, fg_styled + bg_image)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = to_tensor(fg_styled + bg_image).float()

    img = transforms.ToPILImage()(img)
    img.save("prova.png")

    print("SAVED: %s" % out_fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--img_fname", type=str, default="", help="path of image to style")
    parser.add_argument("-o", "--out_fname", type=str, default="", help="path of image to style")
    parser.add_argument("-s", "--style", type=int, default=0, help="Choose a style 0-3")
    parser.add_argument("-c", "--class_id", type=int, default=11, help="Choose a class_id 1-20")
    args = parser.parse_args()

    main(args.img_fname, args.out_fname, args.style, args.class_id)