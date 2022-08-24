# III. Model documentation and writeup

### 1. Who are you (mini-bio) and what do you do professionally?

I'm from Ukraine, I work as a freelance data scientist. I regularly participate in various online competitions.

### 2. What motivated you to compete in this challenge?

I took part in weather-themed competitions, so I decided to take part in this competition as well, having some experience with the topic.

### 3. High level summary of your approach: what did you do and why?

The data consist of a variety of data sources. To do this, we decided to use a neural network model with different layer architectures: Fourier neural operator (FNO), convolution layers, embeddings, and linear transformation. This architecture allows us to combine data of different nature: images, historical data and classification labels. Also FNO can decrease the influence of errors in the data.

### 4. Do you have any useful charts, graphs, or visualizations from the process?

See model report pdf file.

### 5. Copy and paste the 3 most impactful parts of your code and explain what each does and how it helped your model.

Data normalization:

```
    band = xr.concat([
            (ds.t00 - 273.15) / 20,
            (ds.t12 - 273.15) / 20,
            (ds.sdwe**0.25 - 1),
            (ds.pwat - 8) / 7,
            ds.refc / 10,
            ds.u / 20,
            ds.v / 20,
            ds.sdwea,
            ds.NDSI.ffill('time').fillna(0).reduce(np.nanmean, ("x", "y")),
            (ds.sd / 200) - 3.6,
        ], dim = 'feature'
    )
```

Crossfold model:

```
    models = []
    for fold_idx in range(5):
        model = SnowNet(features=10, h_dim=64, width=92, timelag=92)
        model.load_state_dict(
            torch.load(f'{args.model_dir}/SnowNet_fold_{fold_idx}_last.pt')['model']
        )
        models.append(model)
    model = ModelAggregator(models)
```

Load static data:

```
    dem = torch.from_numpy(images_dem / 1000 - 2.25).float().unsqueeze(1)
    soil = torch.from_numpy(images_soil).long()
```

### 6. Please provide the machine specs and time you used to run your model.

- CPU (model): Ryzen 5600X
- GPU (model or N/A): N/A
- Memory (GB): 16Gb
- OS: Ubuntu
- Train duration: ~ 2-3h
- Inference duration: 20 min (aws t3 instance)

### 7. Anything we should watch out for or be aware of in using your model (e.g. code quirks, memory requirements, numerical stability issues, etc.)?

Models are simple. Looks closely at the data preparation part.

### 8. Did you use any tools for data preparation or exploratory data analysis that aren’t listed in your code submission?

No.

### 9. How did you evaluate performance of the model other than the provided metric, if at all?

I only used cross-validation with RMSE.

### 10. What are some other things you tried that didn’t necessarily make it into the final workflow (quick overview)?

I tried many different weather parameters, many of them reduced the RMSE, increasing the number of parameters resulted in rapid overfitting.

### 11. If you were to continue working on this problem for the next year, what methods or techniques might you try in order to build on your work so far? Are there other fields or features you felt would have been very helpful to have?

Better model performance requires more data, especially from different locations.