# Annax: Approximate Nearest Neighbor Search with JAX

Annax is a high-performance Approximate Nearest Neighbor (ANN) library built on top of the JAX framework. It provides fast and memory-efficient search for high-dimensional data in various applications, such as large-scale machine learning, computer vision, and natural language processing. Annax leverages the power of GPU acceleration to deliver outstanding performance and includes a wide range of indexing structures and distance metrics to cater to different use cases. The easy-to-use API makes it accessible to both beginners and experts in the field.

## Features

- Fast and memory-efficient approximate nearest neighbor search
- GPU acceleration for high-performance computing
- Supports a wide range of indexing structures and distance metrics
- Easy-to-use API for seamless integration with existing projects
- Applicable to various domains, including machine learning, computer vision, and natural language processing
- Built on top of the JAX framework for enhanced flexibility and extensibility

## Installation

To install Annax, simply run the following command in your terminal:

```bash
pip install annax
```

## Quick Start

Here's a simple example of using Annax to find the nearest neighbors in a dataset:

```python
import numpy as np
import annax

# Generate some random high-dimensional data
data = np.random.random((1000, 128))

# Create an Annax index with the default configuration
index = annax.Index(data)

# Query for the 10 nearest neighbors of a random vector
query = np.random.random(128)
neighbors, distances = index.search(query, k=10)
```

## Development

To install Annax for development, run the following commands in your terminal:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

## License

Annax is released under the MIT License.
