import torch
import torch.optim as optim
import torch.nn.functional as F
from train_utils import create_model,sample_noise, gan_checkpoint, cyclegan_checkpoint
from utils.data_utils import to_var, gan_save_samples, cyclegan_save_samples, create_dir, label_ones, label_zeros
from utils.data_loader import get_emoji_loader

def gan_training_loop(dataloader, test_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    # Create generators and discriminators
    G, D = create_model(opts)
    g_params = G.parameters()  # Get generator parameters
    d_params = D.parameters()  # Get discriminator parameters
    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr * 2., [opts.beta1, opts.beta2])

    train_iter = iter(dataloader)
    test_iter = iter(test_dataloader)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_noise = sample_noise(100, opts.noise_size)  # 100 x noise_size x 1 x 1
    iter_per_epoch = len(train_iter)
    total_train_iters = opts.train_iters

    try:
        for iteration in range(1, opts.train_iters + 1):
            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)

            real_images, real_labels = train_iter.next()
            real_images, real_labels = to_var(real_images), to_var(real_labels).long().squeeze()

            d_optimizer.zero_grad()
            # Compute the discriminator loss on real images
            real_labels = label_ones(real_images.size(0))
            D_real_loss = F.mse_loss(D(real_images), real_labels)

            # Sample noise
            noise = sample_noise(real_labels.size(0), opts.noise_size)
            # Generate fake images from the noise
            fake_images = G(noise)
            # Compute the discriminator loss on the fake images
            fake_labels = label_zeros(fake_images.size(0))
            D_fake_loss = F.mse_loss(D(fake_images), fake_labels)
            # Compute the total discriminator loss
            D_total_loss = 0.5 * D_real_loss + 0.5 * D_fake_loss

            D_total_loss.backward()
            d_optimizer.step()

            # TRAIN THE GENERATOR
            g_optimizer.zero_grad()
            # Sample noise
            noise = sample_noise(real_labels.size(0), opts.noise_size)
            # Generate fake images from the noise
            fake_images = G(noise)
            # Compute the generator loss
            real_labels = label_ones(fake_images.size(0))
            G_loss = F.mse_loss(D(fake_images), real_labels)

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                    iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
            # Save the generated samples
            if iteration % opts.sample_every == 0:
                gan_save_samples(G, fixed_noise, iteration, opts)
            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                gan_checkpoint(iteration, G, D, opts)

    except KeyboardInterrupt:
        print('Exiting early from training.')
        return G, D

    return G, D

def cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = to_var(test_iter_X.next()[0])
    fixed_Y = to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    try:
        for iteration in range(1, opts.train_iters + 1):
            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                iter_X = iter(dataloader_X)
                iter_Y = iter(dataloader_Y)

            images_X, labels_X = iter_X.next()
            images_X, labels_X = to_var(images_X), to_var(labels_X).long().squeeze()

            images_Y, labels_Y = iter_Y.next()
            images_Y, labels_Y = to_var(images_Y), to_var(labels_Y).long().squeeze()

            # TRAIN THE DISCRIMINATORS
            # Train with real images
            d_optimizer.zero_grad()
            # Compute the discriminator losses on real images
            real_labels = label_ones(images_X.size(0))
            D_X_loss = F.mse_loss(D_X(images_X), real_labels)
            real_labels = label_ones(images_Y.size(0))
            D_Y_loss = F.mse_loss(D_Y(images_Y), real_labels)
            d_real_loss = D_X_loss + D_Y_loss

            d_real_loss.backward()
            d_optimizer.step()

            # Train with fake images
            d_optimizer.zero_grad()

            # Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)
            # Compute the loss for D_X
            fake_labels = label_zeros(fake_X.size(0))
            D_X_loss = F.mse_loss(D_X(fake_X), fake_labels)

            # Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)
            # Compute the loss for D_Y
            fake_labels = label_zeros(fake_Y.size(0))
            D_Y_loss = F.mse_loss(D_Y(fake_Y), fake_labels)

            d_fake_loss = D_X_loss + D_Y_loss
            d_fake_loss.backward()
            d_optimizer.step()

            # TRAIN THE GENERATORS
            # Y--X-->Y CYCLE
            g_optimizer.zero_grad()

            # Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)
            # Compute the generator loss based on domain X
            fake_labels = label_ones(fake_X.size(0))
            g_loss = F.mse_loss(D_X(fake_X), fake_labels)

            reconstructed_Y = G_XtoY(fake_X)
            # Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.abs(reconstructed_Y - images_Y).sum() / images_Y.size(0)
            g_loss += opts.lambda_cycle * cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

            # X--Y-->X CYCLE
            g_optimizer.zero_grad()

            # Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)
            # Compute the generator loss based on domain Y
            fake_labels = label_ones(fake_Y.size(0))
            g_loss = F.mse_loss(D_Y(fake_Y), fake_labels)

            reconstructed_X = G_YtoX(fake_Y)
            # Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.abs(reconstructed_X - images_X).sum() / images_X.size(0)
            # cycle_consistency_loss = F.mse_loss(reconstructed_X, images_X)
            g_loss += opts.lambda_cycle * cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                      'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(), D_Y_loss.item(),
                    D_X_loss.item(), d_fake_loss.item(), g_loss.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                cyclegan_save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                cyclegan_checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)

    except KeyboardInterrupt:
        print('Exiting early from training.')
        return G_XtoY, G_YtoX, D_X, D_Y

    return G_XtoY, G_YtoX, D_X, D_Y

def train(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    if opts.Y:
        dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    # Start training
    if opts.Y is None:
        G, D = gan_training_loop(dataloader_X, test_dataloader_X, opts)
        return G, D
    else:
        G_XtoY, G_YtoX, D_X, D_Y = cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X,
                                                          test_dataloader_Y, opts)
        return G_XtoY, G_YtoX, D_X, D_Y


