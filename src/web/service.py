# 服务类
class TitleService:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, text: str):
        return self.predictor.predict(text)
