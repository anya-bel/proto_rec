import torch

from algo import rnn_rnnat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#possible values for the first lang: fr, rom, sp, por, it
input_lang, output_lang, pairs, test_pairs = rnn_rnnat.prepareData('/data/romance-ortography.txt', 'fr', 'latin')
hidden_size = 256
encoder1 = rnn_rnnat.EncoderRNN(input_lang.n_letters, hidden_size).to(device)
attn_decoder1 = rnn_rnnat.AttnDecoderRNN(hidden_size, output_lang.n_letters, dropout_p=0.1).to(device)
rnn_rnnat.trainIters(input_lang, output_lang, encoder1, attn_decoder1, pairs=pairs, n_iters=25000, print_every=5000)

print("".join(rnn_rnnat.evaluate(input_lang, output_lang, encoder1, attn_decoder1, 'léopard1')))