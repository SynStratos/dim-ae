# dim_ae
Dimensionality Reduction Autoencoder built with Keras TF


### Package installation
```
pip install dimae
```

### Usage example
```python
import pandas as pd
import tensorflow as tf

from dimae.autoencoders.autoencoder import AE


df = pd.DataFrame(...)

n_features = df.shape[1]
output_features = 10

ae = AE(n_features, output_features)

batch_size = 8
epochs = 15

dataset = tf.data.Dataset.from_tensor_slices((df.values.astype('float32'), df.values.astype('float32')))
t_dataset = dataset.batch(batch_size)

ae.compile(optimizer = 'adam', loss = 'mse')
ae.fit(t_dataset, epochs = epoches)

encoder = ae.generate_encoder()
encoder.summary()
```
