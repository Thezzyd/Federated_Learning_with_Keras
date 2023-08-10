import flwr as fl
import sys
import numpy as np

class NewModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

strategy = NewModelStrategy(min_fit_clients=3, min_eval_clients = 3, min_available_clients=3)

fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config={'num_rounds': 5} ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)