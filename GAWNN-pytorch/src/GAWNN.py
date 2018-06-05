import torch
import numpy as np
from torch.autograd import Variable
from src.layers import (ConvLayer, DeconvLayer, Flatten, GlobalAveragePooling)
from src.positional import PositionTrasnformer


class GeneratorGAWNN(torch.nn.Module):
    def __init__(self, hidden_size, z_size, appearance_size, position_size):
        '''
        
        '''
        super(GeneratorGAWNN, self).__init__()
        self.appearance_encoder = torch.nn.Sequential(
            ConvLayer(3, 32, 3, 2),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 64, 3, 2),
            GlobalAveragePooling(),
            Flatten()
        )

        # transform image into multiple maps
        self.position_transformer = PositionTrasnformer(hidden_size)

        self.position_encoder = torch.nn.Sequential(
            ConvLayer(position_size, 32, 3, 2),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 64, 3, 2),
            GlobalAveragePooling(),
            Flatten()
        )

        self.global_network = torch.nn.Sequential(
            DeconvLayer(64 * 2 + z_size, 64, 3, 2, 0),
            DeconvLayer(64, 32, 3, 2, 0),
            DeconvLayer(32, 16, 3, 2, 0),
            DeconvLayer(16, 16, 2, 1, 0)
        )

        self.local_network = torch.nn.Sequential(
            DeconvLayer(64 * 2 + z_size, 64, 3, 2, 0),
            DeconvLayer(64, 32, 3, 2, 0),
            DeconvLayer(32, 16, 3, 2, 0),
            DeconvLayer(16, 16, 2, 1, 0)
        )

        self.final_layer = torch.nn.Sequential(
            DeconvLayer(16 * 2 + position_size, 128, 3, 2, 0),
            DeconvLayer(128, 64, 3, 2, 1),
            DeconvLayer(64, 3, 2, 2, 1),
            torch.nn.Tanh()
        )


    def forward(self, z, appearance_descriptor, position_descriptor):
        '''
        z - random vector
        appearance_descriptor - image of size 3xSxS
        position_descriptor - vector of key points
        '''
        appearance_representation = \
            self.appearance_encoder(appearance_descriptor)


        transformed_position = self.position_transformer(position_descriptor)
        position_representation = \
            self.position_encoder(
                transformed_position
            )

        hidden_vector = torch.cat([
            z, 
            appearance_representation, 
            position_representation], 1
            )

        hidden_vector = hidden_vector.view(
            hidden_vector.shape[0],
            hidden_vector.shape[1],
            1,
            1
            )

        global_result = self.global_network(hidden_vector)
        #print("Global result shape: ", global_result.shape)
        local_result = self.local_network(hidden_vector)
        local_result = local_result * torch.sum(transformed_position, dim=1, keepdim=True)
        #print("Local result shape: ", global_result.shape)

        final_representation = torch.cat([
            global_result, local_result, transformed_position
        ], 1)
        #print("Transformed position shape: ", transformed_position.shape)
        #print("Final represntation shape: ", final_representation.shape)

        return self.final_layer(final_representation)


class DiscriminatorGAWNN(torch.nn.Module):
    def __init__(self, image_size, appearance_size, position_size, hidden_size):
        super(DiscriminatorGAWNN, self).__init__()
        self.global_network = torch.nn.Sequential(
            ConvLayer(3, 32, 3, 2),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 64, 2, 2, 1),
        )


        self.local_network = torch.nn.Sequential(
            ConvLayer(3, 32, 3, 2),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 64, 2, 2, 1),
        )

        self.appearance_encoder = torch.nn.Sequential(
            ConvLayer(3, 32, 3, 2),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 64, 2, 2, 1),
        )

        self.post_global_network = torch.nn.Sequential(
            ConvLayer(64, 256, 3, 2),
            ConvLayer(256, 512, 3, 2),
            ConvLayer(512, 1024, 3, 2),
            Flatten()
        )

        self.post_local_network = torch.nn.Sequential(
            ConvLayer(146, 256, 3, 2),
            ConvLayer(256, 512, 3, 2),
            ConvLayer(512, 1024, 3, 2),
            Flatten()
        )

        self.final_network = torch.nn.Sequential(
            torch.nn.Linear(2048, 1),
            torch.nn.Sigmoid()
        )

        self.position_transformer = PositionTrasnformer(hidden_size)

    def forward(self, images, appearance, positions):
        appearance_embedding = self.appearance_encoder(appearance)
        local_result = self.local_network(images)
        local_result = torch.cat([local_result, appearance_embedding], dim=1)
        transformed_position = self.position_transformer(positions)

        binary_position_mask = \
            torch.sum(transformed_position, dim=1)[:, None]
        binary_position_mask = binary_position_mask.expand(
            -1, 
            local_result.shape[1],
            -1,
            -1)
        local_result = local_result * binary_position_mask

        local_result = torch.cat([local_result, transformed_position], dim=1)
        local_result = self.post_local_network(local_result)

        global_result = self.global_network(images)
        global_result = self.post_global_network(global_result)
        fin = torch.cat([local_result, global_result], dim=1)
        return self.final_network(fin)
