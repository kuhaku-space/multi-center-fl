import copy

from mcfl.client import Client, ClientUpdate
from mcfl.config import Config
from mcfl.server import ClientEval, Server


class DittoServer(Server):
    """
    Ditto (Li et al., 2021) のサーバー

    グローバルモデルを FedAvg で学習しつつ、各クライアントがグローバルモデルへの近接項
    (λ/2)‖v − w‖² 付きで自分の個別モデル v を学習し、個別モデルで評価する。
    個別モデルはグローバルモデルの初期値から始め、参加したラウンドにだけ更新する
    """

    def __init__(self, config: Config, clients: list[Client]):
        super().__init__(config)
        self.assignments = {c.writer_id: 0 for c in clients}
        self.personal_models = {
            c.writer_id: copy.deepcopy(self.models[0]) for c in clients
        }

    def participate(self, client: Client) -> ClientUpdate:
        """グローバルモデルのローカル学習（FedAvg の更新）と、個別モデルの学習を行う"""
        global_model = self.models[0]
        weights, avg_loss = client.train_from(global_model)
        personal_loss = client.train(
            self.personal_models[client.writer_id],
            self.cfg.local_epochs,
            prox_mu=self.cfg.ditto_lambda,
            anchor_model=global_model,
        )
        scores = {"Personal Loss": personal_loss}
        return ClientUpdate(client.writer_id, 0, 0, scores, weights, avg_loss)

    def evaluate(self, clients: list[Client]) -> list[ClientEval]:
        """各クライアントのテストデータを、そのクライアントの個別モデルで評価する"""
        return [
            ClientEval(
                client.writer_id,
                0,
                *client.evaluate(self.personal_models[client.writer_id]),
            )
            for client in clients
            if client.test_loader is not None
        ]
