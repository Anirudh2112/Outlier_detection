from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class ProstateFAnoGAN:
    def __init__(self, 
                 input_shape=(96, 96, 96, 1),
                 latent_dim=128,  
                 batch_size=4):  
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        # Initialize models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.encoder = self._build_encoder()
        
        # Build WGAN
        self.wgan = self._build_wgan()
        
    def _build_generator(self):
        """Build generator network."""
        noise_input = Input(shape=(self.latent_dim,))
        
        # Initial dense layer and reshape
        x = Dense(6 * 6 * 6 * 256)(noise_input)
        x = Reshape((6, 6, 6, 256))(x)
        
        # Upsampling blocks
        x = self._generator_block(x, 128)  # 12x12x12
        x = self._generator_block(x, 64)   # 24x24x24
        x = self._generator_block(x, 32)   # 48x48x48
        x = self._generator_block(x, 16)   # 96x96x96
        
        # Output layer
        output = Conv3D(1, 3, padding='same', activation='tanh')(x)
        
        return Model(noise_input, output, name='generator')
    
    def _generator_block(self, x, filters):
        x = Conv3DTranspose(filters, 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x
    
    def _build_discriminator(self):
        """Build discriminator network with feature extraction."""
        image_input = Input(shape=self.input_shape)
        
        # Convolution blocks
        x = self._discriminator_block(image_input, 16)  # 48x48x48
        x = self._discriminator_block(x, 32)   # 24x24x24
        x = self._discriminator_block(x, 64)   # 12x12x12
        x = self._discriminator_block(x, 128)  # 6x6x6
        
        # Feature extraction layer
        features = Conv3D(256, 3, padding='same')(x)
        
        flattened = Flatten()(features)
        
        # Output layer for WGAN
        validity = Dense(1)(flattened)
        
        model = Model(image_input, [validity, features], name='discriminator')
        return model
    
    def _discriminator_block(self, x, filters):
        x = Conv3D(filters, 4, strides=2, padding='same')(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x
    
    def _build_encoder(self):
        """Build encoder network."""
        image_input = Input(shape=self.input_shape)
        
        x = self._discriminator_block(image_input, 16)
        x = self._discriminator_block(x, 32)
        x = self._discriminator_block(x, 64)
        x = self._discriminator_block(x, 128)
        
        # Output latent vector
        x = Flatten()(x)
        x = Dense(self.latent_dim, activation='tanh')(x)
        
        return Model(image_input, x, name='encoder')
    
    def _build_wgan(self):
        self.discriminator.trainable = False

        latent = Input(shape=(self.latent_dim,))
        generated_image = self.generator(latent)
        valid, _ = self.discriminator(generated_image)
        
        return Model(latent, valid)

    def _train_discriminator(self, real_batch, batch_size, optimizer, grad_penalty_weight):
        # Generate random latent vectors
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)
            
            # Get discriminator outputs
            disc_real = self.discriminator(real_batch, training=True)
            disc_fake = self.discriminator(generated_images, training=True)

            real_validity, real_features = disc_real
            fake_validity, fake_features = disc_fake
            
            # Wasserstein loss
            d_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity)
            
            # Gradient penalty
            alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
            interpolates = alpha * real_batch + (1 - alpha) * generated_images
            
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolates)
                disc_interpolates = self.discriminator(interpolates, training=True)
                validity_interpolates = disc_interpolates[0]  
            
            gradients = gp_tape.gradient(validity_interpolates, interpolates)
            gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3, 4]))
            gradient_penalty = grad_penalty_weight * tf.reduce_mean(tf.square(gradient_norms - 1))
            
            # Total discriminator loss
            d_loss += gradient_penalty
        
        # Update discriminator weights with gradient clipping
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_gradients = [tf.clip_by_norm(g, 1.0) for g in d_gradients]
        optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return d_loss

    def _train_generator(self, batch_size, optimizer):
        noise = tf.random.normal((batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate images
            generated_images = self.generator(noise, training=True)
            # Get discriminator output
            disc_fake = self.discriminator(generated_images, training=True)
            fake_validity = disc_fake[0]  # Get just the validity score
            # Generator loss
            g_loss = -tf.reduce_mean(fake_validity)
        
        # Update generator weights
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        g_gradients = [tf.clip_by_norm(g, 1.0) for g in g_gradients]
        optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return g_loss

    def _train_encoder_step(self, real_batch, optimizer):
        with tf.GradientTape() as tape:
            # Get latent vectors for real images
            latent = self.encoder(real_batch, training=True)
            # Generate reconstructed images
            reconstructed = self.generator(latent, training=False)
            
            # Get discriminator features
            _, real_features = self.discriminator(real_batch, training=False)
            _, fake_features = self.discriminator(reconstructed, training=False)
            
            epsilon = 1e-8
            
            # Compute losses with clipping
            reconstruction_loss = tf.reduce_mean(tf.clip_by_value(
                tf.abs(real_batch - reconstructed), epsilon, 1e6))
            feature_loss = tf.reduce_mean(tf.clip_by_value(
                tf.abs(real_features - fake_features), epsilon, 1e6))
            
            # Total loss
            total_loss = reconstruction_loss + 0.1 * feature_loss
        
        # Add gradient clipping
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        return total_loss
            
    def train_wgan(self, training_data, n_epochs=20, batch_size=8, n_critic=5):
        if batch_size is None:
            batch_size = self.batch_size
        
        # Training parameters
        grad_penalty_weight = 5.0
        max_gradient_norm = 1.0
        
        # Optimizers with reduced learning rates
        d_optimizer = Adam(learning_rate=0.00001, beta_1=0, beta_2=0.9)
        g_optimizer = Adam(learning_rate=0.00001, beta_1=0, beta_2=0.9)
        
        # Reshape data if needed
        if len(training_data.shape) == 4:
            training_data = training_data[..., np.newaxis]
        
        n_batches = len(training_data) // batch_size
        
        d_losses = []
        g_losses = []
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            epoch_d_losses = []
            epoch_g_losses = []
            
            for batch_idx in range(n_batches):
                # Get real batch
                idx = np.random.randint(0, training_data.shape[0], batch_size)
                real_batch = training_data[idx]
                
                n_critic_current = n_critic * 2 if epoch < 2 else n_critic
                
                # Train discriminator
                for _ in range(n_critic_current):
                    d_loss = self._train_discriminator(real_batch, batch_size, 
                                                    d_optimizer, grad_penalty_weight)
                    epoch_d_losses.append(float(d_loss))
                
                # Train generator
                g_loss = self._train_generator(batch_size, g_optimizer)
                epoch_g_losses.append(float(g_loss))
                
                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx+1}/{n_batches}, "
                        f"D Loss: {float(d_loss):.4f}, "
                        f"G Loss: {float(g_loss):.4f}")

            d_losses.append(np.mean(epoch_d_losses))
            g_losses.append(np.mean(epoch_g_losses))
            
            print(f"Epoch {epoch+1} Average - D Loss: {d_losses[-1]:.4f}, G Loss: {g_losses[-1]:.4f}")
            
            # Early stopping check
            if epoch > 2 and np.std(d_losses[-3:]) > 50:  
                print("Training unstable, reducing learning rate...")
                d_optimizer.learning_rate = d_optimizer.learning_rate * 0.5
                g_optimizer.learning_rate = g_optimizer.learning_rate * 0.5
        
        training_history = {
            'd_losses': d_losses,
            'g_losses': g_losses
        }
        
        print("\nWGAN Training Summary:")
        print("D Loss trend:", [f"{loss:.4f}" for loss in d_losses])
        print("G Loss trend:", [f"{loss:.4f}" for loss in g_losses])
        
        return training_history
    
    def train_encoder(self, training_data, n_epochs=50, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        # Optimizer
        optimizer = Adam(learning_rate=0.0001)
        
        # Reshape data if needed
        if len(training_data.shape) == 4:
            training_data = training_data[..., np.newaxis]
            
        n_batches = len(training_data) // batch_size
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            epoch_losses = []
            
            for batch_idx in range(n_batches):
                idx = np.random.randint(0, training_data.shape[0], batch_size)
                real_batch = training_data[idx]
                
                loss = self._train_encoder_step(real_batch, optimizer)
                epoch_losses.append(loss)
                
                if batch_idx % 5 == 0:
                    print(f"Batch {batch_idx+1}/{n_batches}, Loss: {loss:.4f}")
            
            print(f"Epoch {epoch+1} Average Loss: {np.mean(epoch_losses):.4f}")
    
    def detect_anomalies(self, test_image):
        # Ensure correct shape
        if len(test_image.shape) == 3:  
            test_image = test_image[np.newaxis, ..., np.newaxis]  
        elif len(test_image.shape) == 4:  
            test_image = test_image[..., np.newaxis]
                
        latent = self.encoder.predict(test_image, batch_size=1)
            
        # Generate reconstruction
        reconstruction = self.generator.predict(latent, batch_size=1)
            
        # Get discriminator features
        _, real_features = self.discriminator.predict(test_image, batch_size=1)
        _, fake_features = self.discriminator.predict(reconstruction, batch_size=1)
            
        # Calculate anomaly scores
        recon_error = np.mean((test_image - reconstruction)**2)
        feature_error = np.mean((real_features - fake_features)**2)
            
        anomaly_score = recon_error + feature_error
            
        return {
            'anomaly_score': float(anomaly_score), 
            'reconstruction': reconstruction[0, ..., 0],  
            'recon_error': float(recon_error),
            'feature_error': float(feature_error)
        }


    def save_weights(self, path):
        self.generator.save(f'{path}_generator')
        self.discriminator.save(f'{path}_discriminator')
        self.encoder.save(f'{path}_encoder')

    def load_weights(self, path):
        self.generator = tf.keras.models.load_model(f'{path}_generator')
        self.discriminator = tf.keras.models.load_model(f'{path}_discriminator')
        self.encoder = tf.keras.models.load_model(f'{path}_encoder')

    def _train_encoder_step(self, real_batch, optimizer):
        with tf.GradientTape() as tape:
            # Get latent vectors for real images
            latent = self.encoder(real_batch, training=True)
            
            # Generate reconstructed images
            reconstructed = self.generator(latent, training=False)
            
            # Get discriminator features
            _, real_features = self.discriminator(real_batch, training=False)
            _, fake_features = self.discriminator(reconstructed, training=False)
            
            # Compute losses with numerical stability
            reconstruction_loss = tf.reduce_mean(tf.abs(real_batch - reconstructed) + 1e-6)
            feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features) + 1e-6)
            
            # Total loss with scaled feature loss
            total_loss = reconstruction_loss + 0.01 * feature_loss  
        
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        return total_loss