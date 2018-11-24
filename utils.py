import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['image.cmap'] = 'gray'

def save_images(generator, epoch, i, filename_prefix):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=.05, hspace=.05)
    z = generate_nosie(100)
    images_fake = generator(z)
    images_fake = images_fake.data.data.cpu().numpy()
    for img_num, sample in enumerate(images_fake):
        ax = plt.subplot(gs[img_num])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32), cmap='Greys_r')

    filename = filename_prefix + str(epoch) + "-" + str(i) 
    plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
    plt.close(fig)