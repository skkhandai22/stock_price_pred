import streamlit as st
from streamlit_option_menu import option_menu
from model import *
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Stock Prediction Explorer',page_icon="logo.png",layout='wide',initial_sidebar_state='auto')


if __name__ == '__main__':
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 6, 1])
    st.markdown("""
    <nav class ="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FF4B$B;">
    """,unsafe_allow_html=True)
    with col1:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)
    st.sidebar.title('''**Stock Price Predictive Explorer**''')
    st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"]
    > div:first-child {width: 450px;}[data-testid="stSidebar"][aria-expanded="false"]
    > div:first-child {width: 450px;margin-left: -400px;}</style>""",
    unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader("Upload Data File", type=['csv'], accept_multiple_files=False)

    if uploaded_files:
        # # choice = st.selectbox("Stock Price Prediction Explorer", ("Dataset"),, "Closing Time Series", "Decomposition Plot","Stock Prediction"),
        # choice_dataset = st.checkbox("Dataset",help="Visualising the Stock data with the attributes.")
        # if choice_dataset:
        #     stock1 = pd.read_csv(uploaded_files)
        #     st.dataframe(stock1)
        #
        # choice_dataset = st.checkbox("Closing Stock Time Plot", help="Visualising the Stock data with the attributes.")
        # if choice_dataset:
        #     stock = pd.read_csv(uploaded_files)
        #     stock.Date = pd.to_datetime(stock.Date, format='%Y%m%d', errors='ignore')
        #
        #     cols = ['High', 'Low', 'Open', 'Volume', 'Adj Close']
        #     stock.drop(cols, axis=1, inplace=True)
        #     stock = stock.sort_values('Date')
        #
        #     stock = stock.groupby('Date')['Close'].sum().reset_index()
        #
        #     stock = stock.set_index('Date')
        #     monthly_mean1 = closingtimeseries(stock)
        #     decomposition1 = decomposition(stock)
        #     st.info(
        #         "We will use the averages daily sales value for that month instead, and we are using the start of each month as the timestamp.")
        #     st.info('Visualising closing Time Series Records')
        #     monthly_mean1.plot(figsize=(15, 6))
        #     st.pyplot()
        #     # st.dataframe(stock)
        #
        # choice_dataset = st.checkbox("Decomposition Plot", help="Visualising the Stock data with the attributes.")
        # if choice_dataset:
        #     stock = pd.read_csv(uploaded_files)
        #     st.dataframe(stock)
        #
        # choice_dataset = st.checkbox("Stock Prediction for Nect 5 years", help="Visualising the Stock data with the attributes.")
        # if choice_dataset:
        #     stock = pd.read_csv(uploaded_files)
        #     st.dataframe(stock)

        selected = option_menu(
            menu_title="",
            options=["Dataset","Closing Time Series","Decomposition Plot","Stock Prediction"],
            orientation="horizontal"
        )
        stock=pd.read_csv(uploaded_files)
        if selected=="Dataset":
            st.info('Dataset')
            st.dataframe(stock)

        # stock.head()
        # stock['Date'].min()
        # stock['Date'].max()
        stock.Date = pd.to_datetime(stock.Date, format='%Y%m%d', errors='ignore')

        cols = ['High', 'Low', 'Open', 'Volume', 'Adj Close']
        stock.drop(cols, axis=1, inplace=True)
        stock = stock.sort_values('Date')


        stock = stock.groupby('Date')['Close'].sum().reset_index()


        stock = stock.set_index('Date')
        monthly_mean1 = closingtimeseries(stock)
        decomposition1 = decomposition(stock)

        if selected=="Closing Time Series":
            st.info("We will use the averages daily sales value for that month instead, and we are using the start of each month as the timestamp.")
            st.info('Visualising closing Time Series Records')
            monthly_mean1.plot(figsize=(15, 6))
            st.pyplot()

        # decomposiotion=decomposition_plot(monthly_mean)

        if selected=="Decomposition Plot":
            st.info("Some distinguishable patterns appear when we plot the data. The time-series has seasonality pattern. We can visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise")
            st.info("Decomposition Plot")
            decomposition1.plot()
            st.pyplot()

        # results,mod=arima(monthly_mean)

        # if selected=="ARIMA Prediction":
        #     monthly_mean, decomposition, results, mod, pred, pred_uc, pred_ci = plots(stock)
        #     st.info("ARIMA Prediction")
        #     results = mod.fit()
        #     # print(results.summary().tables[1])
        #     results.plot_diagnostics(figsize=(16, 8))
        #     st.pyplot()

        if selected=="Stock Prediction":
            monthly_mean, decomposition, results, mod, pred, pred_uc, pred_ci = plots(stock)
            st.info("Validating forecast")
            # pred,pred_ci=forecast(results,monthly_mean)
            ax = monthly_mean['2014':].plot(label='observed')
            pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.2)

            ax.set_xlabel('Date')
            ax.set_ylabel('close price')
            plt.legend()

            plt.show()
            st.pyplot()

            st.info("Stock Predictions for next 5 years")
            # pred_uc,pred_ci=actual_pred(results,monthly_mean,pred)
            ax = monthly_mean.plot(label='observed', figsize=(14, 7))
            pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.25)
            ax.set_xlabel('Date')
            ax.set_ylabel('close price')

            plt.legend()
            plt.show()
            st.pyplot()




