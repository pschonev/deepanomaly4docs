import plotly.express as px


def load_coords_to_df(df, coords_2d):
    if "X" not in df and "Y" not in df:
        df["X"] = coords_2d[:, 0]
        df["Y"] = coords_2d[:, 1]

    return df


def prepare_text(df, col, line_chars=75):
    textlen = line_chars * 30
    df["htext"] = df["text"].str.replace(r'\\n', '<br>', regex=True)
    df["htext"] = df["htext"].map(lambda x: "<br>".join(
        x[i:i+line_chars] for i in range(0, len(x), line_chars)))
    df["htext"] = df["htext"].str[0: textlen]

    df["char_count"] = df["text"].apply(len)

    return df


def create_show_graph(df, col, coords_2d=None, color="title", line_chars=75, kwargs={}):
    df = load_coords_to_df(df, coords_2d)
    df = prepare_text(df, col)

    default_kwargs = {'x':'X', 'y':'Y', 'color':"title", 'hover_data':["title", "htext", "char_count"],
                     'color_discrete_sequence':px.colors.qualitative.Dark24, 'color_discrete_map':{"-1": "rgb(255, 255, 255)"}}
    default_kwargs.update(kwargs)

    print("Create graph ...")
    fig = px.scatter(df, **default_kwargs)
    fig.show()
