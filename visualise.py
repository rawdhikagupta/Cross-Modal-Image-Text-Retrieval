#DATA VISUALISATION
###################################################
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

def visualize_sample(image, caption):
    # Denormalize the image
    transform = transforms.Compose([
        transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
    ])
    denormalized_image = transform(image.clone().detach()).numpy().transpose(1, 2, 0)

    # Display the image
    plt.imshow(denormalized_image)
    plt.axis('off')
    # Display the caption on the image
    
    caption_text = " ".join(caption)
    plt.text(0, 0, caption_text, color='white', backgroundcolor='black', fontsize=8, verticalalignment='top')

    plt.show()

if __name__ == '__main__':
    # Your existing code here
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    main('/Users/radhikagupta/Downloads/SOP', 'rsicd_precomp')

    # Load or deserialize the vocabulary
    vocab = Vocabulary()
    vocab_path = '/Users/radhikagupta/Downloads/SOP/rsitmd_splits_vocab.json'  # Update with the correct path
    vocab = deserialize_vocab(vocab_path)

    # Assuming you have a function get_loaders in your script
    opt = {}  # Update with your actual optional parameters
    train_loader, val_loader = get_loaders(vocab, opt)

    # Visualize a sample from the training loader
    count = 1; 
    for batch in train_loader:
        images, caption, _, _ = batch
        for i in range(len(images)):
            visualize_sample(images[i], [vocab.idx2word[str(idx.item())] for idx in caption[i] if idx != 0])
            break  # Display only the first sample for brevity
        count += 1
        if count>10:
            break  # Display only the first batch for brevity

###################################################

