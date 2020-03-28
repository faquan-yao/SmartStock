

def segment(model, data_x, data_y, Threshold, minseg=2):
    if len(data) <= minseg:
        return

    y_pre = model.predict(data_x)
