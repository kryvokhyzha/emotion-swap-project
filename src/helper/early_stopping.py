class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


if __name__ == "__main__":
    early_stopping = EarlyStopping(tolerance=2, min_delta=5)

    train_loss = [
        642.14990234,
        601.29278564,
        561.98400879,
        530.01501465,
        497.1098938,
        466.92709351,
        438.2364502,
        413.76028442,
        391.5090332,
        370.79074097,
    ]
    validate_loss = [
        509.13619995,
        497.3125,
        506.17315674,
        497.68960571,
        505.69918823,
        459.78610229,
        480.25592041,
        418.08630371,
        446.42675781,
        372.09902954,
    ]

    for i in range(len(train_loss)):

        early_stopping(train_loss[i], validate_loss[i])
        print(f"loss: {train_loss[i]} : {validate_loss[i]}")
        if early_stopping.early_stop:
            print("We are at epoch:", i)
            break
