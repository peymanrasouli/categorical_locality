from sklearn.metrics import f1_score, accuracy_score

def CreateModel(X_train, X_test, Y_train, Y_test, model_name, constructor):
        if model_name is 'svm':
                blackbox = constructor(random_state=42, probability=True)
        else:
                blackbox = constructor(random_state=42)
        blackbox.fit(X_train, Y_train)
        pred_test = blackbox.predict(X_test)
        bb_accuracy_score = accuracy_score(Y_test, pred_test)
        print(model_name , 'blackbox accuracy=', bb_accuracy_score)
        bb_f1_score = f1_score(Y_test, pred_test, average='weighted')
        print(model_name , 'blackbox F1-score=', bb_f1_score)
        return blackbox


