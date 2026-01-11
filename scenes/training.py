"""
Training stage comparison scenes for Encoder-Decoder vs Decoder-Only transformers.
Phase 3: Training Stage Comparison
"""

from manim import *
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.transformer_block import TransformerBlock
from components.attention_matrix import AttentionMatrix


class EncoderDecoderTraining(Scene):
    """
    Task 3.1: Encoder-Decoder Training Flow

    Example: Translating 'Hello World' → '你好 世界'

    Shows:
    1. Input sequence enters Encoder (bidirectional processing)
    2. Target sequence shifted right (<BOS> at start) as Decoder input
    3. Teacher Forcing: training Decoder input is real target tokens
    4. Parallel computation: one forward pass for all positions
    5. Cross-Attention querying Encoder output
    """

    def construct(self):
        # Title
        title = Text("Encoder-Decoder Training", font_size=32, color=WHITE)
        subtitle = Text("Translation: 'Hello World' → '你好 世界'", font_size=20, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.5)

        # === ENCODER SIDE ===
        encoder_label = Text("Encoder", font_size=22, color=BLUE)
        encoder_label.to_edge(LEFT, buff=1.0).shift(UP * 2)

        # Input tokens
        input_tokens_text = ["Hello", "World", "<EOS>"]
        input_tokens = VGroup()
        for txt in input_tokens_text:
            token = VGroup(
                RoundedRectangle(width=1.0, height=0.6, corner_radius=0.1, color=BLUE_C, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            input_tokens.add(token)

        input_tokens.arrange(RIGHT, buff=0.15)
        input_tokens.next_to(encoder_label, DOWN, buff=0.5)

        # Encoder block (simplified)
        encoder_block = VGroup(
            RoundedRectangle(width=3.5, height=1.5, corner_radius=0.1, color=BLUE, fill_opacity=0.2),
            Text("Bidirectional\nSelf-Attention", font_size=14, color=BLUE)
        )
        encoder_block[1].move_to(encoder_block[0].get_center())
        encoder_block.next_to(input_tokens, DOWN, buff=0.3)

        # Encoder output (hidden states)
        encoder_output = VGroup()
        for i in range(3):
            state = VGroup(
                RoundedRectangle(width=0.7, height=0.5, corner_radius=0.05, color=GREEN, fill_opacity=0.4),
                Text(f"h{i+1}", font_size=12, color=WHITE)
            )
            state[1].move_to(state[0].get_center())
            encoder_output.add(state)

        encoder_output.arrange(RIGHT, buff=0.15)
        encoder_output.next_to(encoder_block, DOWN, buff=0.3)

        encoder_states_label = Text("Encoder States", font_size=12, color=GREEN)
        encoder_states_label.next_to(encoder_output, DOWN, buff=0.1)

        # Bidirectional arrows (show each token can see all others)
        bi_arrows = VGroup()
        arrow_positions = [
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
        ]

        # Animate Encoder
        self.play(FadeIn(encoder_label))
        self.play(
            *[FadeIn(token, shift=DOWN * 0.3) for token in input_tokens],
            lag_ratio=0.2
        )

        # Show bidirectional attention
        bi_attention_note = Text("Each token sees all tokens", font_size=12, color=BLUE_C)
        bi_attention_note.next_to(input_tokens, UP, buff=0.1)
        self.play(FadeIn(bi_attention_note))

        self.play(FadeIn(encoder_block))
        self.play(
            *[FadeIn(state, shift=DOWN * 0.2) for state in encoder_output],
            FadeIn(encoder_states_label),
            lag_ratio=0.15
        )
        self.wait(0.5)

        # === DECODER SIDE ===
        decoder_label = Text("Decoder", font_size=22, color=ORANGE)
        decoder_label.to_edge(RIGHT, buff=1.5).shift(UP * 2)

        # Target tokens (shifted right - Teacher Forcing)
        target_tokens_text = ["<BOS>", "你好", "世界"]
        target_tokens = VGroup()
        for txt in target_tokens_text:
            token = VGroup(
                RoundedRectangle(width=0.9, height=0.6, corner_radius=0.1, color=ORANGE, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            target_tokens.add(token)

        target_tokens.arrange(RIGHT, buff=0.15)
        target_tokens.next_to(decoder_label, DOWN, buff=0.5)

        # Teacher Forcing label
        tf_label = Text("Teacher Forcing:\nReal target tokens as input", font_size=11, color=YELLOW)
        tf_label.next_to(target_tokens, UP, buff=0.1)

        # Shifted right indicator
        shift_label = Text("(shifted right)", font_size=10, color=GRAY)
        shift_label.next_to(target_tokens, DOWN, buff=0.05)

        # Decoder block with two attention mechanisms
        decoder_block = VGroup(
            RoundedRectangle(width=3.8, height=2.2, corner_radius=0.1, color=ORANGE, fill_opacity=0.2),
        )

        masked_attn = VGroup(
            RoundedRectangle(width=3.2, height=0.5, corner_radius=0.05, color=YELLOW, fill_opacity=0.3),
            Text("Masked Self-Attention", font_size=11, color=YELLOW)
        )
        masked_attn[1].move_to(masked_attn[0].get_center())

        cross_attn = VGroup(
            RoundedRectangle(width=3.2, height=0.5, corner_radius=0.05, color=GREEN, fill_opacity=0.3),
            Text("Cross-Attention", font_size=11, color=GREEN)
        )
        cross_attn[1].move_to(cross_attn[0].get_center())

        ffn = VGroup(
            RoundedRectangle(width=3.2, height=0.4, corner_radius=0.05, color=PURPLE, fill_opacity=0.3),
            Text("FFN", font_size=11, color=PURPLE)
        )
        ffn[1].move_to(ffn[0].get_center())

        decoder_internals = VGroup(masked_attn, cross_attn, ffn).arrange(DOWN, buff=0.15)
        decoder_internals.move_to(decoder_block[0].get_center())
        decoder_block.add(decoder_internals)
        decoder_block.next_to(shift_label, DOWN, buff=0.3)

        # Cross-attention connection from Encoder
        cross_arrow = CurvedArrow(
            start_point=encoder_output.get_right() + RIGHT * 0.2,
            end_point=cross_attn.get_left() + LEFT * 0.2,
            color=GREEN,
            stroke_width=2,
            angle=-TAU/8,
        )
        kv_label = Text("K, V", font_size=12, color=GREEN)
        kv_label.move_to(cross_arrow.get_center() + UP * 0.3)

        # Q label from decoder
        q_label = Text("Q", font_size=12, color=ORANGE)
        q_label.next_to(cross_attn, RIGHT, buff=0.1)

        # Decoder output (predictions)
        output_tokens_text = ["你好", "世界", "<EOS>"]
        output_tokens = VGroup()
        for txt in output_tokens_text:
            token = VGroup(
                RoundedRectangle(width=0.9, height=0.6, corner_radius=0.1, color=YELLOW, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            output_tokens.add(token)

        output_tokens.arrange(RIGHT, buff=0.15)
        output_tokens.next_to(decoder_block, DOWN, buff=0.3)

        output_label = Text("Predictions (parallel)", font_size=12, color=YELLOW)
        output_label.next_to(output_tokens, DOWN, buff=0.1)

        # Animate Decoder
        self.play(FadeIn(decoder_label))
        self.play(
            *[FadeIn(token, shift=DOWN * 0.3) for token in target_tokens],
            lag_ratio=0.2
        )
        self.play(FadeIn(tf_label), FadeIn(shift_label))
        self.wait(0.3)

        self.play(FadeIn(decoder_block))

        # Show Cross-Attention connection
        self.play(
            Create(cross_arrow),
            FadeIn(kv_label),
            FadeIn(q_label),
        )
        self.wait(0.5)

        # Show parallel output
        parallel_note = Text("All positions computed in parallel", font_size=12, color=YELLOW)
        parallel_note.to_edge(DOWN, buff=1.5)

        self.play(
            *[FadeIn(token, shift=DOWN * 0.2) for token in output_tokens],
            FadeIn(output_label),
            run_time=1.0
        )
        self.play(FadeIn(parallel_note))
        self.wait(0.5)

        # Loss computation visualization
        loss_arrows = VGroup()
        for i in range(3):
            arrow = Arrow(
                output_tokens[i].get_bottom() + DOWN * 0.1,
                output_tokens[i].get_bottom() + DOWN * 0.6,
                color=RED,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.3,
            )
            loss_arrows.add(arrow)

        loss_label = Text("Cross-Entropy Loss", font_size=14, color=RED)
        loss_label.next_to(loss_arrows, DOWN, buff=0.1)

        self.play(
            *[Create(arrow) for arrow in loss_arrows],
            FadeIn(loss_label),
        )
        self.wait(0.5)

        # Summary
        summary = VGroup(
            Text("Key Points:", font_size=16, color=WHITE),
            Text("1. Encoder processes input bidirectionally", font_size=13, color=BLUE),
            Text("2. Decoder receives shifted targets (Teacher Forcing)", font_size=13, color=ORANGE),
            Text("3. Cross-Attention: Q from Decoder, K/V from Encoder", font_size=13, color=GREEN),
            Text("4. All output positions computed in parallel", font_size=13, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        summary.to_edge(DOWN, buff=0.2)

        self.play(
            FadeOut(parallel_note),
            FadeOut(loss_label),
            FadeOut(loss_arrows),
        )
        self.play(FadeIn(summary))
        self.wait(3)


class DecoderOnlyTraining(Scene):
    """
    Task 3.2: Decoder-Only Training Flow

    Example: Predicting 'The cat sat on the mat'

    Shows:
    1. Causal Mask animation (building lower triangular matrix)
    2. Each position can only attend to itself and previous tokens
    3. Parallel training: one forward produces all position predictions
    4. Loss computed at each position (predicting next token)
    """

    def construct(self):
        # Title
        title = Text("Decoder-Only Training", font_size=32, color=WHITE)
        subtitle = Text("Next Token Prediction: 'The cat sat on the'", font_size=20, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.5)

        # Input sequence
        input_label = Text("Input Sequence", font_size=18, color=BLUE_C)
        input_label.shift(LEFT * 4 + UP * 1.5)

        tokens_text = ["The", "cat", "sat", "on", "the"]
        input_tokens = VGroup()
        for i, txt in enumerate(tokens_text):
            token = VGroup(
                RoundedRectangle(width=0.8, height=0.55, corner_radius=0.1, color=BLUE_C, fill_opacity=0.3),
                Text(txt, font_size=13, color=WHITE),
                Text(f"t{i+1}", font_size=10, color=GRAY)
            )
            token[1].move_to(token[0].get_center())
            token[2].next_to(token[0], DOWN, buff=0.05)
            input_tokens.add(token)

        input_tokens.arrange(RIGHT, buff=0.1)
        input_tokens.next_to(input_label, DOWN, buff=0.3)

        self.play(FadeIn(input_label))
        self.play(
            *[FadeIn(token, shift=RIGHT * 0.2) for token in input_tokens],
            lag_ratio=0.15
        )
        self.wait(0.3)

        # === CAUSAL MASK VISUALIZATION ===
        mask_title = Text("Causal Mask", font_size=18, color=ORANGE)
        mask_title.shift(RIGHT * 3.5 + UP * 1.8)

        # Create causal mask matrix
        causal_matrix = AttentionMatrix(
            size=5,
            tokens=["t1", "t2", "t3", "t4", "t5"],
            attention_type="causal",
            cell_size=0.45,
        )
        causal_matrix.scale(0.85)
        causal_matrix.shift(RIGHT * 3.5 + DOWN * 0.3)

        self.play(FadeIn(mask_title))

        # Initially show empty matrix then build it
        # First show the grid outline
        empty_matrix = AttentionMatrix(
            size=5,
            tokens=["t1", "t2", "t3", "t4", "t5"],
            attention_type="bidirectional",
            cell_size=0.45,
        )
        empty_matrix.scale(0.85)
        empty_matrix.shift(RIGHT * 3.5 + DOWN * 0.3)

        # Set all cells to masked initially
        for i in range(5):
            for j in range(5):
                cell = empty_matrix.get_cell(i, j)
                cell.set_fill(GREY_D, opacity=0.2)
                cell.set_stroke(GREY_D)

        self.play(FadeIn(empty_matrix))
        self.wait(0.3)

        # Animate building causal mask row by row
        mask_explanation = Text("", font_size=12, color=GREEN)
        mask_explanation.to_edge(RIGHT, buff=0.5).shift(DOWN * 2.5)

        for i in range(5):
            cells_to_highlight = []
            for j in range(i + 1):
                cell = empty_matrix.get_cell(i, j)
                cells_to_highlight.append(cell)

            explanation_text = f"'{tokens_text[i]}' sees: {', '.join(tokens_text[:i+1])}"
            new_explanation = Text(explanation_text, font_size=11, color=GREEN)
            new_explanation.to_edge(RIGHT, buff=0.3).shift(DOWN * 2.5)

            self.play(
                *[cell.animate.set_fill(YELLOW, opacity=0.7).set_stroke(YELLOW) for cell in cells_to_highlight],
                Transform(mask_explanation, new_explanation) if i > 0 else FadeIn(new_explanation),
                run_time=0.4
            )
            if i == 0:
                mask_explanation = new_explanation
            self.wait(0.2)

        self.wait(0.5)

        # === DECODER PROCESSING ===
        decoder_label = Text("Decoder (Causal Self-Attention)", font_size=16, color=ORANGE)
        decoder_label.shift(LEFT * 3 + DOWN * 0.5)

        decoder_block = VGroup(
            RoundedRectangle(width=4.5, height=1.2, corner_radius=0.1, color=ORANGE, fill_opacity=0.2),
            Text("Causal Self-Attention + FFN", font_size=12, color=ORANGE)
        )
        decoder_block[1].move_to(decoder_block[0].get_center())
        decoder_block.next_to(decoder_label, DOWN, buff=0.2)

        self.play(FadeIn(decoder_label), FadeIn(decoder_block))

        # Output predictions
        output_label = Text("Predictions (parallel)", font_size=14, color=YELLOW)
        output_label.next_to(decoder_block, DOWN, buff=0.4)

        pred_tokens_text = ["cat", "sat", "on", "the", "mat"]
        output_tokens = VGroup()
        for i, txt in enumerate(pred_tokens_text):
            token = VGroup(
                RoundedRectangle(width=0.8, height=0.55, corner_radius=0.1, color=YELLOW, fill_opacity=0.3),
                Text(txt, font_size=13, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            output_tokens.add(token)

        output_tokens.arrange(RIGHT, buff=0.1)
        output_tokens.next_to(output_label, DOWN, buff=0.2)

        # Show parallel computation
        parallel_note = Text("One forward pass → all predictions", font_size=12, color=YELLOW)
        parallel_note.next_to(output_tokens, DOWN, buff=0.3)

        self.play(FadeIn(output_label))
        self.play(
            *[FadeIn(token, shift=DOWN * 0.2) for token in output_tokens],
            run_time=0.8
        )
        self.play(FadeIn(parallel_note))
        self.wait(0.5)

        # Loss computation at each position
        loss_label = Text("Loss at each position", font_size=14, color=RED)
        loss_label.next_to(parallel_note, DOWN, buff=0.4)

        # Show loss arrows connecting input to output
        loss_connections = VGroup()
        for i in range(5):
            start = input_tokens[i].get_bottom() + DOWN * 0.1
            end = output_tokens[i].get_top() + UP * 0.1

            # Curved arrow
            connection = CurvedArrow(
                start_point=start + DOWN * 0.5,
                end_point=end + UP * 0.3,
                color=RED,
                stroke_width=1.5,
                angle=-TAU/6,
            )
            loss_connections.add(connection)

        self.play(FadeIn(loss_label))
        self.play(
            *[Create(conn) for conn in loss_connections],
            run_time=1.0
        )

        # Explain loss
        loss_explanation = Text("Each position predicts the next token", font_size=11, color=RED)
        loss_explanation.next_to(loss_label, DOWN, buff=0.1)
        self.play(FadeIn(loss_explanation))
        self.wait(0.5)

        # Summary
        summary = VGroup(
            Text("Key Points:", font_size=14, color=WHITE),
            Text("1. Causal mask: each position sees only past tokens", font_size=11, color=ORANGE),
            Text("2. All positions processed in parallel (training)", font_size=11, color=YELLOW),
            Text("3. Loss: predict next token at each position", font_size=11, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        summary.to_edge(DOWN, buff=0.15)

        self.play(
            FadeOut(loss_connections),
            FadeOut(loss_explanation),
        )
        self.play(FadeIn(summary))
        self.wait(3)


class TrainingComparison(Scene):
    """
    Task 3.3: Training Comparison Summary

    Side-by-side comparison of training methods:
    - Same: Both use Teacher Forcing, both can train in parallel
    - Different: Encoder-Decoder has bidirectional encoding + Cross-Attention
    """

    def construct(self):
        # Title
        title = Text("Training Comparison", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

        # Create two sides
        left_title = Text("Encoder-Decoder", font_size=22, color=BLUE)
        right_title = Text("Decoder-Only", font_size=22, color=ORANGE)

        left_title.shift(LEFT * 3.5 + UP * 2)
        right_title.shift(RIGHT * 3.5 + UP * 2)

        self.play(FadeIn(left_title), FadeIn(right_title))

        # === LEFT SIDE: Encoder-Decoder Training ===
        # Simplified diagram
        enc_input = VGroup(
            Text("Input:", font_size=12, color=GRAY),
            VGroup(*[
                RoundedRectangle(width=0.6, height=0.4, color=BLUE_C, fill_opacity=0.3)
                for _ in range(3)
            ]).arrange(RIGHT, buff=0.05)
        ).arrange(DOWN, buff=0.1)

        enc_box = VGroup(
            RoundedRectangle(width=2.5, height=0.8, color=BLUE, fill_opacity=0.2),
            Text("Encoder\n(Bidirectional)", font_size=10, color=BLUE)
        )
        enc_box[1].move_to(enc_box[0].get_center())

        dec_input = VGroup(
            Text("Target:", font_size=12, color=GRAY),
            VGroup(*[
                RoundedRectangle(width=0.6, height=0.4, color=ORANGE, fill_opacity=0.3)
                for _ in range(3)
            ]).arrange(RIGHT, buff=0.05)
        ).arrange(DOWN, buff=0.1)

        dec_box = VGroup(
            RoundedRectangle(width=2.5, height=1.0, color=ORANGE, fill_opacity=0.2),
            Text("Decoder\n(Masked + Cross)", font_size=10, color=ORANGE)
        )
        dec_box[1].move_to(dec_box[0].get_center())

        enc_output = VGroup(*[
            RoundedRectangle(width=0.5, height=0.35, color=GREEN, fill_opacity=0.4)
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.05)

        left_diagram = VGroup(enc_input, enc_box, enc_output, dec_input, dec_box)
        left_diagram.arrange(DOWN, buff=0.2)
        left_diagram.next_to(left_title, DOWN, buff=0.3)

        # Cross-attention arrow
        cross_arrow = Arrow(
            enc_output.get_right(),
            dec_box.get_left() + UP * 0.2,
            color=GREEN,
            stroke_width=2,
        )

        self.play(FadeIn(left_diagram), Create(cross_arrow))

        # === RIGHT SIDE: Decoder-Only Training ===
        dec_only_input = VGroup(
            Text("Input:", font_size=12, color=GRAY),
            VGroup(*[
                RoundedRectangle(width=0.55, height=0.4, color=ORANGE, fill_opacity=0.3)
                for _ in range(5)
            ]).arrange(RIGHT, buff=0.03)
        ).arrange(DOWN, buff=0.1)

        dec_only_box = VGroup(
            RoundedRectangle(width=3.2, height=1.0, color=ORANGE, fill_opacity=0.2),
            Text("Decoder\n(Causal Self-Attention)", font_size=10, color=ORANGE)
        )
        dec_only_box[1].move_to(dec_only_box[0].get_center())

        # Small causal mask indicator
        mask_indicator = AttentionMatrix(
            size=4,
            tokens=["", "", "", ""],
            attention_type="causal",
            cell_size=0.2,
            show_labels=False,
        )
        mask_indicator.scale(0.6)

        dec_only_output = VGroup(*[
            RoundedRectangle(width=0.55, height=0.4, color=YELLOW, fill_opacity=0.3)
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.03)

        right_diagram = VGroup(dec_only_input, dec_only_box, mask_indicator, dec_only_output)
        right_diagram.arrange(DOWN, buff=0.2)
        right_diagram.next_to(right_title, DOWN, buff=0.3)

        self.play(FadeIn(right_diagram))
        self.wait(0.5)

        # Comparison table
        table_title = Text("Comparison", font_size=20, color=WHITE)
        table_title.to_edge(DOWN, buff=2.2)

        # Similarities
        similarities = VGroup(
            Text("Similarities:", font_size=16, color=GREEN),
            Text("✓ Both use Teacher Forcing", font_size=13, color=WHITE),
            Text("✓ Both train in parallel", font_size=13, color=WHITE),
            Text("✓ Both compute loss at all positions", font_size=13, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        similarities.shift(LEFT * 3.5 + DOWN * 1.8)

        # Differences
        differences = VGroup(
            Text("Differences:", font_size=16, color=RED),
            Text("• Enc-Dec: Bidirectional encoder", font_size=13, color=BLUE),
            Text("• Enc-Dec: Cross-Attention", font_size=13, color=BLUE),
            Text("• Dec-Only: Causal attention only", font_size=13, color=ORANGE),
            Text("• Dec-Only: Single sequence", font_size=13, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        differences.shift(RIGHT * 3 + DOWN * 1.7)

        self.play(FadeIn(similarities))
        self.wait(0.5)
        self.play(FadeIn(differences))
        self.wait(3)
