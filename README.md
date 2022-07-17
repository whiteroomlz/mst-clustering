# Spanning tree clustering

For installation use next command:

```bash
    pip install git+https://github.com/whiteroomlz/spanning-tree-clustering.git
```

The class `SpanningTreeClustering` uses the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) 
module, so you should create an entry point in your main script:

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ...
```
