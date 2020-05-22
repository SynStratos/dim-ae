from autoencoders.autoencoder import AE
import numpy as np
a = AE(n_input = 1000, code_nodes = 800, summary=False, _n_layers= 3)
# a.fit()
encoder = a.generate_encoder()
encoder.summary()
