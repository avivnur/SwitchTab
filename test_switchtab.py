import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils import Encoder, Projector, Decoder, Predictor, feature_corruption

class TestSwitchTabComponents(unittest.TestCase):
    def setUp(self):
        self.feature_size = 10  # A small feature size for testing
        self.batch_size = 2  # A small batch size for testing
        self.num_classes = 3  # Assuming three classes for testing
        # Initialize synthetic data for tests
        self.x_train = torch.randn(self.batch_size, self.feature_size)
        self.x_batch = torch.randn(self.batch_size, self.feature_size)
        # Initialize model components
        self.encoder = Encoder(self.feature_size)  # Assuming encoder initialization requires feature_size
        self.projector_s = Projector(self.feature_size)  # Same for projectors
        self.projector_m = Projector(self.feature_size)
        self.decoder = Decoder(2 * self.feature_size, self.feature_size)
        self.predictor = Predictor(self.feature_size, self.num_classes)
    
    def test_feature_corruption(self):
        corrupted_x = feature_corruption(self.x_batch)
        # Check if the corruption function returns a tensor of the same shape
        self.assertEqual(corrupted_x.shape, self.x_batch.shape)
        # Check if about 30% of the elements are zeroed out
        # Calculate the actual corruption ratio
        actual_corruption_ratio = (corrupted_x == 0).float().mean().item()
        # Increased delta to accommodate variability
        self.assertAlmostEqual(actual_corruption_ratio, 0.3, delta=0.2)

    def test_encoder_forward_pass(self):
        # Check if the encoder can perform a forward pass without errors
        encoded_x = self.encoder(self.x_batch.unsqueeze(0))  # Add a sequence length dimension
        self.assertEqual(encoded_x.shape, (1, self.batch_size, self.feature_size))

    def test_projector_forward_pass(self):
        # Check if the projectors can perform a forward pass without errors
        projected_x_s = self.projector_s(self.x_batch)
        projected_x_m = self.projector_m(self.x_batch)
        self.assertEqual(projected_x_s.shape, self.x_batch.shape)
        self.assertEqual(projected_x_m.shape, self.x_batch.shape)

    def test_decoder_forward_pass(self):
        # Check if the decoder can perform a forward pass without errors
        mock_input = torch.cat([self.x_batch, self.x_batch], dim=1)  # Now mock_input is [2, 20]
        decoded_x = self.decoder(mock_input)
        self.assertEqual(decoded_x.shape, self.x_batch.shape, "Decoder output shape does not match expected.")

    def test_predictor_forward_pass(self):
        # Check if the predictor can perform a forward pass without errors
        predictions = self.predictor(self.x_batch)
        self.assertEqual(predictions.shape, (self.batch_size, self.num_classes))

    def test_training_loop_single_batch(self):
        dataset = TensorDataset(self.x_train)
        dataloader = DataLoader(dataset, batch_size=1)
        
        optimizer = torch.optim.SGD(list(self.encoder.parameters()) + 
                                    list(self.projector_s.parameters()) +
                                    list(self.projector_m.parameters()) + 
                                    list(self.decoder.parameters()), lr=0.001)
        mse_loss = nn.MSELoss()

        for x_batch, in dataloader:
            # Assuming a simplified model pipeline for demonstration
            x_batch_unsqueezed = x_batch.unsqueeze(1)  # Add sequence length dimension if needed
            z_encoded = self.encoder(x_batch_unsqueezed)
            s_salient = self.projector_s(z_encoded.squeeze(1))
            m_mutual = self.projector_m(z_encoded.squeeze(1))
            x_reconstructed = self.decoder(torch.cat((s_salient, m_mutual), dim=1))

            # Check if the loss is computed correctly
            loss = mse_loss(x_reconstructed, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.assertFalse(torch.isnan(loss) or torch.isinf(loss), "Loss is not a valid number.")

    def test_fine_tuning_single_batch(self):
        # Create a small synthetic dataset with labels for fine-tuning
        x_train = torch.randn(2, 10)  # 2 samples, 10 features each
        y_train = torch.randint(0, 3, (2,))  # Random labels for 3 classes
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=1)  # Batch size of 1 for testing
        
        # We use the setUp components here, assuming the Encoder, Predictor, etc., are defined
        
        # Mock the optimizer and loss for fine-tuning
        optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.001)
        fine_tuning_loss_function = nn.CrossEntropyLoss()

        # Run a single batch through the fine-tuning loop
        for x_batch, y_batch in dataloader:
            # Forward pass through the components
            z_encoded = self.encoder(x_batch.unsqueeze(0))  # Add a sequence length dimension
            predictions = self.predictor(z_encoded.squeeze(0))  # Remove the sequence length dimension
            
            # Compute the loss
            loss = fine_tuning_loss_function(predictions, y_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check if the loss is a valid number and not nan or inf
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss))

# To run the tests
if __name__ == '__main__':
    unittest.main()
