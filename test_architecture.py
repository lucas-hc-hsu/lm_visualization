"""Test file for architecture scenes."""

from manim import *
import sys
sys.path.insert(0, '/home/hhc_wsl/lm_visualization')

from components.transformer_block import TransformerBlock
from components.attention_matrix import AttentionMatrix


class EncoderDecoderArchitecture(Scene):
    """
    Task 2.1: Encoder-Decoder Architecture Animation
    """

    def construct(self):
        # Title
        title = Text("Encoder-Decoder Architecture", font_size=36, color=WHITE)
        subtitle = Text("(T5, BART, mBART)", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.5)

        # Create Encoder side
        encoder_title = Text("Encoder", font_size=24, color=BLUE)

        # Encoder blocks (stack of 3)
        encoder_blocks = VGroup()
        for i in range(3):
            block = TransformerBlock(
                block_type="encoder",
                width=2.5,
                height=2.2,
                show_residual=True,
            )
            encoder_blocks.add(block)

        encoder_blocks.arrange(UP, buff=0.3)
        encoder_title.next_to(encoder_blocks, UP, buff=0.2)

        # Nx indicator
        encoder_nx = Text("×N", font_size=20, color=BLUE)
        encoder_nx.next_to(encoder_blocks, RIGHT, buff=0.1)

        encoder_group = VGroup(encoder_blocks, encoder_title, encoder_nx)

        # Create Decoder side
        decoder_title = Text("Decoder", font_size=24, color=ORANGE)

        # Decoder blocks (stack of 3 with cross-attention)
        decoder_blocks = VGroup()
        for i in range(3):
            block = TransformerBlock(
                block_type="enc_dec_decoder",
                width=2.5,
                height=3.2,
                show_residual=True,
            )
            decoder_blocks.add(block)

        decoder_blocks.arrange(UP, buff=0.3)
        decoder_title.next_to(decoder_blocks, UP, buff=0.2)

        # Nx indicator
        decoder_nx = Text("×N", font_size=20, color=ORANGE)
        decoder_nx.next_to(decoder_blocks, RIGHT, buff=0.1)

        decoder_group = VGroup(decoder_blocks, decoder_title, decoder_nx)

        # Position encoder and decoder
        encoder_group.shift(LEFT * 3.5 + DOWN * 0.5)
        decoder_group.shift(RIGHT * 3.5 + DOWN * 0.5)

        # Input labels
        encoder_input = Text("Input\nEmbedding", font_size=16, color=BLUE_C)
        encoder_input.next_to(encoder_blocks, DOWN, buff=0.3)

        decoder_input = Text("Output\nEmbedding\n(shifted right)", font_size=14, color=ORANGE)
        decoder_input.next_to(decoder_blocks, DOWN, buff=0.3)

        # Positional Encoding indicators
        pos_enc_left = Text("+ Pos Enc", font_size=14, color=GREEN)
        pos_enc_left.next_to(encoder_input, DOWN, buff=0.1)

        pos_enc_right = Text("+ Pos Enc", font_size=14, color=GREEN)
        pos_enc_right.next_to(decoder_input, DOWN, buff=0.1)

        # Output
        output_label = Text("Output\nProbabilities", font_size=16, color=YELLOW)
        output_label.next_to(decoder_blocks, UP, buff=0.8)

        output_arrow = Arrow(
            start=decoder_blocks.get_top() + UP * 0.1,
            end=output_label.get_bottom() + DOWN * 0.1,
            color=YELLOW,
            stroke_width=2,
        )

        # Animate Encoder
        self.play(
            FadeIn(encoder_blocks),
            Write(encoder_title),
            FadeIn(encoder_nx),
            run_time=1.5
        )
        self.play(
            FadeIn(encoder_input),
            FadeIn(pos_enc_left),
        )
        self.wait(0.5)

        # Animate Decoder
        self.play(
            FadeIn(decoder_blocks),
            Write(decoder_title),
            FadeIn(decoder_nx),
            run_time=1.5
        )
        self.play(
            FadeIn(decoder_input),
            FadeIn(pos_enc_right),
        )
        self.wait(0.5)

        # Create Cross-Attention connections
        cross_attention_arrows = VGroup()
        encoder_top = encoder_blocks.get_top()

        for i, dec_block in enumerate(decoder_blocks):
            cross_attn_pos = dec_block.get_left() + LEFT * 0.3
            arrow = CurvedArrow(
                start_point=encoder_top + RIGHT * 0.5 + DOWN * (0.5 + i * 0.3),
                end_point=cross_attn_pos,
                color=GREEN,
                stroke_width=2,
                angle=-TAU/6,
            )
            cross_attention_arrows.add(arrow)

        # K,V label
        kv_label = Text("K, V", font_size=18, color=GREEN)
        kv_label.move_to((encoder_top + decoder_blocks[1].get_left()) / 2 + UP * 0.5)

        self.play(
            *[Create(arrow) for arrow in cross_attention_arrows],
            FadeIn(kv_label),
            run_time=1.5
        )
        self.wait(0.5)

        # Output arrow
        self.play(
            Create(output_arrow),
            FadeIn(output_label),
        )
        self.wait(0.5)

        # Explanation
        explanation = VGroup(
            Text("• Encoder: Bidirectional Self-Attention", font_size=16, color=BLUE),
            Text("• Decoder: Masked Self-Attention + Cross-Attention", font_size=16, color=ORANGE),
            Text("• Cross-Attention: Q from Decoder, K/V from Encoder", font_size=16, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        explanation.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(explanation))
        self.wait(3)


class DecoderOnlyArchitecture(Scene):
    """
    Task 2.2: Decoder-Only Architecture Animation
    """

    def construct(self):
        # Title
        title = Text("Decoder-Only Architecture", font_size=36, color=WHITE)
        subtitle = Text("(GPT, LLaMA, Mistral)", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.5)

        # Create Decoder blocks (stack of 4)
        decoder_title = Text("Decoder Stack", font_size=24, color=ORANGE)

        decoder_blocks = VGroup()
        for i in range(4):
            block = TransformerBlock(
                block_type="decoder_only",
                width=3.0,
                height=2.5,
                show_residual=True,
            )
            decoder_blocks.add(block)

        decoder_blocks.arrange(UP, buff=0.25)
        decoder_title.next_to(decoder_blocks, UP, buff=0.2)

        # Nx indicator
        decoder_nx = Text("×N", font_size=20, color=ORANGE)
        decoder_nx.next_to(decoder_blocks, RIGHT, buff=0.15)

        decoder_group = VGroup(decoder_blocks, decoder_title, decoder_nx)
        decoder_group.move_to(ORIGIN).shift(DOWN * 0.3)

        # Input label
        input_label = Text("Input Tokens", font_size=18, color=BLUE_C)
        input_label.next_to(decoder_blocks, DOWN, buff=0.4)

        # Positional Encoding
        pos_enc = Text("+ Positional Encoding", font_size=14, color=GREEN)
        pos_enc.next_to(input_label, DOWN, buff=0.1)

        # Input sequence visualization
        input_tokens = VGroup()
        token_texts = ["The", "cat", "sat", "..."]
        for i, txt in enumerate(token_texts):
            token = VGroup(
                RoundedRectangle(width=0.8, height=0.5, corner_radius=0.1, color=BLUE_C, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            input_tokens.add(token)

        input_tokens.arrange(RIGHT, buff=0.1)
        input_tokens.next_to(pos_enc, DOWN, buff=0.2)

        # Output label
        output_label = Text("Next Token Prediction", font_size=18, color=YELLOW)
        output_label.next_to(decoder_blocks, UP, buff=0.5)

        output_arrow = Arrow(
            start=decoder_blocks.get_top() + UP * 0.1,
            end=output_label.get_bottom() + DOWN * 0.1,
            color=YELLOW,
            stroke_width=2,
        )

        # Output tokens (shifted by one)
        output_tokens = VGroup()
        output_texts = ["cat", "sat", "on", "..."]
        for i, txt in enumerate(output_texts):
            token = VGroup(
                RoundedRectangle(width=0.8, height=0.5, corner_radius=0.1, color=YELLOW, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            output_tokens.add(token)

        output_tokens.arrange(RIGHT, buff=0.1)
        output_tokens.next_to(output_label, UP, buff=0.2)

        # Animate decoder stack
        self.play(
            FadeIn(decoder_blocks),
            Write(decoder_title),
            FadeIn(decoder_nx),
            run_time=1.5
        )
        self.wait(0.5)

        # Animate input
        self.play(
            FadeIn(input_label),
            FadeIn(pos_enc),
        )
        self.play(
            *[FadeIn(token, shift=UP * 0.3) for token in input_tokens],
            lag_ratio=0.2,
        )
        self.wait(0.5)

        # Animate output
        self.play(
            Create(output_arrow),
            FadeIn(output_label),
        )
        self.play(
            *[FadeIn(token, shift=UP * 0.3) for token in output_tokens],
            lag_ratio=0.2,
        )
        self.wait(0.5)

        # Show causal mask visualization on the side
        causal_matrix = AttentionMatrix(
            size=4,
            tokens=["t1", "t2", "t3", "t4"],
            attention_type="causal",
            cell_size=0.4,
        )
        causal_matrix.scale(0.8)
        causal_matrix.to_edge(RIGHT, buff=0.5)

        mask_label = Text("Causal Mask", font_size=16, color=ORANGE)
        mask_label.next_to(causal_matrix, UP, buff=0.2)

        self.play(
            FadeIn(causal_matrix),
            FadeIn(mask_label),
        )
        self.wait(0.5)

        # Explanation
        explanation = VGroup(
            Text("• Single stack of decoder blocks", font_size=16, color=ORANGE),
            Text("• Causal Self-Attention only (no Cross-Attention)", font_size=16, color=YELLOW),
            Text("• Input and output form a continuous sequence", font_size=16, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        explanation.to_edge(DOWN, buff=0.3).shift(LEFT * 2)

        self.play(FadeIn(explanation))
        self.wait(3)
