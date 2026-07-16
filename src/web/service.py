# 服务类
class TitleService:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, title):
        return self.predict_text(title.text)

    def predict_text(self, text: str):
        return self.predictor.predict(text)
