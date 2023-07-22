import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

model = pickle.load(open('C:/Users/DELL/PROJECT DS/TIME SERIES FORECASTING/fbprophet.pickle', 'rb'))


def main():
    st.title('Oil Prices Forecasting')

    tsf = pd.read_excel('C:\\Users\\DELL\\OneDrive\\Desktop\\VIVEK DS PROJECTS\\Brent crude oil(daily).xlsx',
                        parse_dates=True)

    tsf.columns = ['ds', 'y']
    tsf['ds'] = pd.to_datetime(tsf['ds'])

    fbtrain = tsf.iloc[:8304]
    fbtest = tsf.iloc[8304:]

    m1 = Prophet()
    m1.fit(fbtrain)

    forecast_period = st.slider("Select forecast period (in days)", min_value=10, max_value=365 * 5, value=365 * 2)

    future1 = m1.make_future_dataframe(periods=forecast_period, freq='D')
    forecast1 = m1.predict(future1)

    st.subheader("Forecast Plot")
    fig = m1.plot(forecast1)
    st.pyplot(fig)

    st.subheader("Components")
    fig = m1.plot_components(forecast1)
    st.pyplot(fig)

    st.write('Forecast Results:')
    st.write(forecast1)

    st.subheader("Cross Validation")

    # Initial 5 years training period
    initial = 5 * 365
    initial = str(initial) + ' days'
    # Fold every 5 years
    period = 5 * 365
    period = str(period) + ' days'
    # Forecast 1 year into the future
    horizon = 365
    horizon = str(horizon) + ' days'

    df_cv = cross_validation(m1, initial=initial, period=period, horizon=horizon)
    st.write('Cross-validation Results:')
    st.write(df_cv)

    available_metrics = ['mse', 'rmse', 'mae', 'mape']
    selected_metric = st.selectbox("Select metric", available_metrics)

    fig = plot_cross_validation_metric(df_cv, metric=selected_metric)
    st.pyplot(fig)

if __name__ == '__main__':
    main()