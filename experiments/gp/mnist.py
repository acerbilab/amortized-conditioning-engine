import hydra
from utils import load_config_and_model, load_config_and_model_tnpd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch
import copy
from src.model.utils import AttrDict
from src.dataset.latents.image_no_np import Image
import matplotlib.pyplot as plt

path = "results/discrete/trained/image/00-54-30/"
path_tnpd = "results/discrete/trained/image/01-07-40/"
config_path = path+".hydra/"

torch.manual_seed(45)

# Load ACE model and config
cfg, model = load_config_and_model(path, path + ".hydra/", 
                                   ckpt_name="ckpt_140000.tar")

# Load TNPD model and config
cfg_tnpd, model_tnpd = load_config_and_model_tnpd(path_tnpd, path_tnpd +".hydra/", 
                                                  ckpt_name="ckpt_140000.tar")


def generate_image(b_xyd, b_xyl, ind, num_ctx):
    xyc = b_xyd[:, :num_ctx, :]
    xyt = b_xyd[:, num_ctx:, :] 

    batch = AttrDict() 
    batch.xc = copy.deepcopy(xyc[:, :, :-1])
    batch.yc = copy.deepcopy(xyc[:, :, -1:])

    batch.xt = copy.deepcopy(xyt[:, :, :-1])
    batch.yt = copy.deepcopy(xyt[:, :, -1:])

    xyc_latent = torch.concat((xyc, b_xyl), dim=1)

    batch_l = AttrDict({
        'xc': xyc_latent[:, :, :-1],
        'yc': xyc_latent[:, :, -1:],
        'xt': xyt[:, :, :-1],
        'yt': xyt[:, :, -1:]
    })

    # First image
    xc = (15 * batch.xc[ind, :, 1:3]).to(torch.long)
    yc = batch.yc[ind, :, -1]
    image_size = 16

    im_context1 = np.zeros((image_size, image_size, 3))
    im_context1[:, :, 2] = 1
    im_context1[xc[:, 0], xc[:, 1]] = yc.repeat(3, 1).transpose(0, 1).cpu()

    # Second image
    xc = (15 * batch.xt[ind, :, 1:3]).to(torch.long)
    yc = batch.yt[ind, :, -1]
    im_context2 = np.zeros((image_size, image_size, 3))
    im_context2[:, :, 2] = 1
    im_context2[xc[:, 0], xc[:, 1]] = yc.repeat(3, 1).transpose(0, 1).cpu()

    out = model.forward(batch, predict=True)
    xc = (15 * batch.xt[ind,:, 1:3]).to(torch.long)
    #yc = out.mean[ind, :, -1]
    yc = out.median[ind, :]
    image_size = 16

    im_context_nl = np.zeros((image_size, image_size, 3))
    im_context_nl[:, :, 2] = 1
    im_context_nl[xc[:, 0], xc[:, 1]] = yc.repeat(3, 1).transpose(0, 1)

    out = model.forward(batch_l, predict=True)
    xc = (15 * batch.xt[ind,:, 1:3]).to(torch.long)
    #yc = out.mean[ind, :, -1]
    yc = out.median[ind, :]
    image_size = 16

    im_context_l = np.zeros((image_size, image_size, 3))
    im_context_l[:, :, 2] = 1
    im_context_l[xc[:, 0], xc[:, 1]] = yc.repeat(3, 1).transpose(0, 1)

    # Plotting the images next to each other
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    axs[0].imshow(im_context1)
    axs[0].axis('off')  # Remove axis labels
    axs[0].set_title('Context Image (10 points)')

    axs[1].imshow(im_context2)
    axs[1].axis('off')  # Remove axis labels
    axs[1].set_title('Target Image')

    axs[2].imshow(im_context_nl)
    axs[2].axis('off')  # Remove axis labels
    axs[2].set_title('ACE no class conditioning')

    axs[3].imshow(im_context_l)
    axs[3].axis('off')  # Remove axis labels
    axs[3].set_title('ACE with class conditioning')

    plt.tight_layout()
    plt.show()


image = Image()
b_xyd, b_xyl = image.get_data() 

num_ctx = 10
ind = 7
#generate_image(b_xyd, b_xyl, ind, num_ctx)

num_ctx = 20
ind = 15
#generate_image(b_xyd, b_xyl, ind, num_ctx)

ctx = [5, 10, 20, 30, 40, 50, 60, 70, 100, 150]
nlpd = np.zeros((len(ctx), 20))
nlpd_l = np.zeros((len(ctx), 20))
nlpd_tnpd = np.zeros((len(ctx), 20))

rmse = np.zeros((len(ctx), 20))
rmse_l = np.zeros((len(ctx), 20))

for j, num_ctx in enumerate(ctx):
    for i in range(20):    
        image = Image()
        b_xyd, b_xyl = image.get_data() 
        xyc = b_xyd[:, :num_ctx, :]
        xyt = b_xyd[:, num_ctx:, :] 

        batch = AttrDict() 
        batch.xc = copy.deepcopy(xyc[:, :, :-1])
        batch.yc = copy.deepcopy(xyc[:, :, -1:])

        batch.xt = copy.deepcopy(xyt[:, :, :-1])
        batch.yt = copy.deepcopy(xyt[:, :, -1:])

        model.eval()
        out = model.forward(batch, predict=True)

        med = out.median[:, :, None]
        square = (batch.yt[:,:,:] - med[:, :, :])**2

        rmse[j, i] = torch.mean(square).item()
        nlpd[j, i] = torch.mean(out.loss).item()

        # Prepare latent batches
        xyc_latent = torch.concat((xyc, b_xyl), dim=1)
        batch_l = AttrDict({
            'xc': xyc_latent[:, :, :-1],
            'yc': xyc_latent[:, :, -1:],
            'xt': xyt[:, :, :-1],
            'yt': xyt[:, :, -1:]
        })

        model.eval()
        out_l = model.forward(batch_l, predict=True)

        med = out_l.median[:, :, None]
        square = (batch.yt[:,:,:] - med[:, :, :])**2
        rmse_l[j, i] = torch.mean(square).item()
        nlpd_l[j, i] = torch.mean(out_l.loss).detach().numpy()

        model_tnpd.eval()
        out_tnpd = model_tnpd.forward(batch)
        nlpd_tnpd[j, i] = torch.mean(out_tnpd.loss).detach().numpy()

# Plot the NLPD
mean_nlpd = np.mean(nlpd, axis=1)
std_nlpd = np.std(nlpd, axis=1)

mean_nlpd_l = np.mean(nlpd_l, axis=1)
std_nlpd_l = np.std(nlpd_l, axis=1)

mean_nlpd_tnpd = np.mean(nlpd_tnpd, axis=1)
std_nlpd_tnpd = np.std(nlpd_tnpd, axis=1)

plt.errorbar(ctx, mean_nlpd, yerr=std_nlpd, fmt='-o', label='ACE no conditioning')
plt.errorbar(ctx, mean_nlpd_l, yerr=std_nlpd, fmt='-o',label='ACE conditioned class')
plt.errorbar(ctx, mean_nlpd_tnpd, yerr=std_nlpd, fmt='-o', label='TNPD')

plt.xlabel('Context Size')
plt.ylabel('NLPD')
plt.title('NLPD vs Context Size')
plt.legend(loc='best')
plt.show()