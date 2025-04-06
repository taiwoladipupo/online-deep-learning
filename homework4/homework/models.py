from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()
        # Define Input dim
        # Each input b has 2 coordinates;
        input_dim = n_track * 4  # 2 sides, each with 2 coordinates
        hidden_dim = 512  # Assuming
        output_dim = n_waypoints * 2  # 2 coordinates for each waypoint

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)

        # Seperate output heads for each coordinate
        self.fc_long = nn.Linear(hidden_dim, n_waypoints)
        # Longitudinal output
        self.fc_lat = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.BatchNorm1d(hidden_dim // 2),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_dim // 2, n_waypoints)
                                    )

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate the left and right track points
        x = torch.cat((track_left, track_right), dim=2)
        # Flatten the input
        x = x.view(x.size(0), -1)  # shape (b, n_track * 4)
        # Pass through the fc1 with relu activation
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Pass through long and lat output heads seperately
        pred_long = self.fc_long(x)
        pred_lat = self.fc_lat(x)

        out = torch.stack((pred_long, pred_lat), dim=2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for each waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        # Linear encoder to project each track point to a higher dimension
        self.transformer_encoder = nn.Linear(4, d_model)
        # Build aa transformer decoder: a single decoder with 4 attention heads

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dropout=0.2)
        # Build the transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Seperate output heads
        self.out_proj_long = nn.Linear(d_model, 1)
        self.out_proj_lat = nn.Linear(d_model, 1)
        #
        # # Final layer
        # self.out_proj = nn.Linear(d_model, 2)

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # concatenate the left and right track points
        track_points = torch.cat((track_left, track_right), dim=2)  # shape (b, n_track, 4)

        # Encode each track points into d_model features
        track_embedded = self.transformer_encoder(track_points)  # shape (b, n_track, d_model)

        # Permute track_embedded to (sequence_length, b, d_model) for transformer decoder
        track_embedded = track_embedded.permute(1, 0, 2)  # shape (n_track, b, d_model)
        # get the query embeddings for the waypoints
        # shape (n_waypoints, b, d_model)
        queries = self.query_embed.weight
        batch_size = track_left.size(0)
        queries = queries.unsqueeze(1).expand(-1, batch_size, -1)  # shape (n_waypoints, b, d_model)

        # Pass through the transformer decoder
        decoded = self.transformer_decoder(queries, track_embedded)

        # Permute the decoded outputs to (b, n_waypoints, d_model)
        decoded = decoded.permute(1, 0, 2)  # shape (b, n_waypoints, d_model)

        # Seperate output heads for long and lat
        pred_long = self.out_proj_long(decoded)
        pred_lat = self.out_proj_lat(decoded)
        # Pass through the final layer to get the waypoints
        waypoints = torch.cat((pred_long, pred_lat), dim=2)
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
            self,
            n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    print(f"Saving model to {output_path.name}")
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
