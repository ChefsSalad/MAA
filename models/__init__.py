# from .model import Generator_gru, Generator_lstm, Generator_transformer, Generator_rnn, Discriminator1, Discriminator2, Discriminator3
# from .model_with_clsHead import Generator_gru, Generator_lstm, Generator_transformer, Generator_rnn, Discriminator3
from .model_with_clsdisc import Generator_gru, Generator_lstm, Generator_transformer, Generator_rnn, Discriminator3

# __all__ = [
#     "Generator_gru",
#     "Generator_lstm",
#     "Generator_transformer",
#     "Generator_rnn",
#     "Discriminator1",
#     "Discriminator2",
#     "Discriminator3"
# ]
__all__ = [
    "Generator_gru",
    "Generator_lstm",
    "Generator_transformer",
    "Generator_rnn",
    "Discriminator3"
]
