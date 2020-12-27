def to_numeric(df, exclude=list(), **kwds):
    for col in df.select_dtypes(exclude=np.number).columns:
        if (True in df[col]) & (False in df[col]):
            #df[col] = df[col].astype('bool')
        else:
            try:
                res = df[col].str.replace("$", "").str.replace(" ", "")
                df[col] = pd.to_numeric(res)
            except:
                pass
        return df
        
def to_category(df, thresh=.05, include = list(),**kwds):
    for col in df.select_dtypes(include='object'):
        ratio = len(df[col].value_counts()) / len(df)
        if ratio < thresh:
            df[col] = df[col].astype('category')

    for col in include:
        try:
            df[col] = df[col].astype('category')
        except:
            pass

    return df

def drop_cols(df, na_thresh=.6, drop_cols=list(), unique_thresh = .05, unique_n=25, cv=.05, **kwds):
    # drop columns with mostly missing
    thresh = len(df) * na_thresh
    df.dropna(axis=1, thresh=thresh, inplace=True)

    # drop columns with lots of values
    for col in df.select_dtypes(exclude=np.number):
        ratio = len(df[col].unique()) / len(df)
        if (ratio > thresh) | (len(df[col].unique()) ==1) | (len(df[col].unique()) > unique_n):
            df.drop(columns = col, inplace=True)


    # drop low variance columns
    for col in df.select_dtypes(include=np.number):
        cv = df[col].mean(skipna=[np.nan,None])/ df[col].std(skipna=[np.nan,None]) 

    # drop user specificed columns
    for col in drop_cols:
        try:
            df.drop(columns=col, inplace=True)
        except:
            pass

    
    return df

def clean_df(df,**kwds):
    df_copy = df.copy()
    df_copy =(df_copy.
               pipe(to_numeric, ).
               pipe(to_category, ).
               pipe(drop_cols, **kwds)
               )
    return df_copy

