def plot_box_per_column(df, columns, ratio_IQR = 1.5):
    import matplotlib.pyplot as plt
    plt.figure()
    for idx, col in enumerate(columns):
        plt.subplot(2,len(columns), idx+1)
        df.boxplot(column=col)

    ##################################################################################################
    ## IQR 기준으로 outlier 지우기: df.loc[df[col].between(df[col].quantile(q=0.25)-1.5*col_IQR, df[col].quantile(q=0.75)+1.5*col_IQR)]
    ##################################################################################################

    print(f"before: length is {len(df)}")
    for col in columns:
        col_IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        df=df.loc[df[col].between(df[col].quantile(q=0.25)-ratio_IQR*col_IQR, df[col].quantile(q=0.75)+ratio_IQR*col_IQR)]

    print(f"after: length is {len(df)}")

    for idx, col in enumerate(columns):
        plt.subplot(2,len(columns), len(columns)+idx+1)
        df.boxplot(column=col)

    plt.show()