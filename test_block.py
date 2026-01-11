"""Test file for TransformerBlock component."""

from manim import *
from components.transformer_block import TransformerBlock


class TestTransformerBlocks(Scene):
    """Test scene to render all three transformer block types."""

    def construct(self):
        # Title
        title = Text("Transformer Block Types", font_size=36)
        title.to_edge(UP)
        self.add(title)

        # Create three block types with smaller width to fit on screen
        encoder_block = TransformerBlock(
            block_type="encoder",
            label="Encoder Block",
            width=2.8,
            height=3.5,
        )

        enc_dec_decoder_block = TransformerBlock(
            block_type="enc_dec_decoder",
            label="Enc-Dec Decoder Block",
            width=2.8,
            height=4.5,
        )

        decoder_only_block = TransformerBlock(
            block_type="decoder_only",
            label="Decoder-Only Block",
            width=2.8,
            height=3.5,
        )

        # Arrange horizontally with more space
        blocks = VGroup(encoder_block, enc_dec_decoder_block, decoder_only_block)
        blocks.arrange(RIGHT, buff=0.8)
        blocks.move_to(ORIGIN).shift(DOWN * 0.3)

        # Scale down if needed to fit
        if blocks.width > 13:
            blocks.scale_to_fit_width(13)

        # Animate
        self.play(FadeIn(encoder_block))
        self.wait(0.5)
        self.play(FadeIn(enc_dec_decoder_block))
        self.wait(0.5)
        self.play(FadeIn(decoder_only_block))
        self.wait(2)
