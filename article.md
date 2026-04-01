# TimesFM for Time Series Forecasting in Python

using Oil Production Data TimesFM removes the need for constant model retraining. It generalizes
across industries. It scales to massive datasets. It represents the...

::::### **TimesFM for Time Series Forecasting in Python using Oil Production Data** 

#### **TimesFM removes the need for constant model retraining. It generalizes across industries. It scales to massive datasets. It represents the next generation of forecasting AI.**
Time series forecasting has always relied on domain-specific models.
Traditional statistical methods like ARIMA and deep learning approaches
like LSTMs work well for structured datasets. But large, diverse time
series --- spanning industries and regions --- need a more generalizable
approach. Google's TimesFM is a foundation model that generalizes across
domains, predicting trends in energy, finance, weather, and industrial
production. Unlike task-specific models, it learns patterns from vast
datasets, applying knowledge across different applications.

This project uses TimesFM to forecast oil production in North Dakota.

Traditional forecasting models require careful tuning. ARIMA works well
for stationary data but struggles with sudden shifts. LSTMs capture
sequences but require extensive training. Transformer-based models like
N-BEATS and PatchTST improve accuracy, but they still train on specific
datasets.

TimesFM solves this by pretraining on diverse time series data. It
learns fundamental patterns, applying them to new datasets without
retraining. This allows faster, more scalable forecasting.

We have monthly production data for about 40,000 wells in North Dakota.
We start by loading the data and selecting the top 2 high-production
wells. This ensures that the forecast focuses on active sites with
meaningful trends.


We split the data into training and test sets using TimeSeriesSplit.
This ensures that the model only sees past data during training.


Now we initialize TimesFM, loading a pretrained checkpoint from Hugging
Face. This gives access to a large-scale foundation model trained on
millions of time series.

We use TimesFM to generate forecasts. The model processes the training
set, making predictions for the test period.


Finally, we visualize the results, comparing actual production with the
TimesFM forecast.



<figcaption>This graph shows just one well instead
of 10,000.</figcaption>


### **Evaluating the Performance of TimesFM**
TimesFM forecasts match the overall trend of oil production. The model
generalizes well, even with limited training data. This confirms that
pretrained time series models can outperform traditional methods with
minimal adaptation.

It does not replace domain-specific models entirely. But it provides a
strong baseline for forecasting without feature engineering.

Because it is a foundation model, it does not require retraining for
each task. It learns universal patterns from massive datasets. This
gives businesses a ready-to-use forecasting engine that scales across
domains.

Foundation models have transformed natural language processing. The same
shift is happening in time series forecasting. TimesFM marks the
beginning of large-scale, cross-domain forecasting models.
::::Update (2025--11--04) I reran TimesFM on a different dataset of [US
Electricity
Generation.](https://www.eia.gov/electricity/data/browser/)
