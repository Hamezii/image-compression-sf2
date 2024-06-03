# This file must import these three names. You are free to move the functions between files, this
# structure is just what we recommend you start with.
from competition.encoder import encode, header_bits
from competition.decoder import decode

if __name__ == '__main__':
    # For convenience only; you can put what you like here, it will not be executed in the
    # competition.
    # See `competition/README.md` for more information.
    from cued_sf2_lab import competition_runner
    competition_runner.main(__name__, imgs=['lighthouse', 'bridge'])