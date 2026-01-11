"""
Summary and integration scenes for Encoder-Decoder vs Decoder-Only transformers.
Phase 5: Integration and Summary
"""

from manim import *
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.transformer_block import TransformerBlock
from components.attention_matrix import AttentionMatrix


class SummaryComparisonTable(Scene):
    """
    Task 5.1: Summary Comparison Table

    Dynamically builds a comparison table:

    | Feature          | Encoder-Decoder        | Decoder-Only          |
    |------------------|------------------------|-----------------------|
    | Architecture     | Encoder + Decoder      | Decoder only          |
    | Attention        | Bi + Causal + Cross    | Causal only           |
    | Input/Output     | Separate sequences     | Continuous sequence   |
    | Training         | Two-stage processing   | Single-stage          |
    | Inference Cache  | Enc states + Dec KV    | Single KV Cache       |
    | Representative   | T5, BART, mBART        | GPT, LLaMA, Mistral   |
    """

    def construct(self):
        # Title
        title = Text("Summary: Encoder-Decoder vs Decoder-Only", font_size=30, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

        # Table data
        rows = [
            ["Feature", "Encoder-Decoder", "Decoder-Only"],
            ["Architecture", "Encoder + Decoder", "Decoder only"],
            ["Attention", "Bi + Causal + Cross", "Causal only"],
            ["Input/Output", "Separate sequences", "Continuous sequence"],
            ["Training", "Two-stage", "Single-stage"],
            ["Inference Cache", "Enc + Dec KV", "Single KV Cache"],
            ["Models", "T5, BART, mBART", "GPT, LLaMA, Mistral"],
        ]

        # Create table cells manually for animation
        cell_width = 3.2
        cell_height = 0.5
        header_height = 0.6

        all_cells = VGroup()
        all_texts = VGroup()

        for i, row in enumerate(rows):
            for j, cell_text in enumerate(row):
                # Determine cell properties
                if i == 0:  # Header row
                    fill_color = GRAY
                    fill_opacity = 0.3
                    text_color = WHITE
                    height = header_height
                elif j == 0:  # First column (feature names)
                    fill_color = GRAY
                    fill_opacity = 0.15
                    text_color = WHITE
                    height = cell_height
                elif j == 1:  # Encoder-Decoder column
                    fill_color = BLUE
                    fill_opacity = 0.15
                    text_color = BLUE
                    height = cell_height
                else:  # Decoder-Only column
                    fill_color = ORANGE
                    fill_opacity = 0.15
                    text_color = ORANGE
                    height = cell_height

                cell = Rectangle(
                    width=cell_width,
                    height=height,
                    color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_width=1,
                )

                x_pos = (j - 1) * cell_width
                y_pos = -i * cell_height + 1.5

                cell.move_to(np.array([x_pos, y_pos, 0]))
                all_cells.add(cell)

                # Text
                text = Text(cell_text, font_size=13, color=text_color)
                text.move_to(cell.get_center())
                all_texts.add(text)

        # Center the table
        table_group = VGroup(all_cells, all_texts)
        table_group.move_to(ORIGIN).shift(DOWN * 0.3)

        # Animate row by row
        num_cols = 3
        for i in range(len(rows)):
            row_cells = VGroup(*[all_cells[i * num_cols + j] for j in range(num_cols)])
            row_texts = VGroup(*[all_texts[i * num_cols + j] for j in range(num_cols)])

            self.play(
                *[Create(cell) for cell in row_cells],
                *[FadeIn(text) for text in row_texts],
                run_time=0.5
            )
            self.wait(0.2)

        self.wait(1)

        # Highlight key differences
        highlight_note = Text(
            "Key: Encoder-Decoder excels at seq2seq tasks, Decoder-Only for generative tasks",
            font_size=12,
            color=GREEN
        )
        highlight_note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(highlight_note))

        self.wait(3)


class FinalIntegration(Scene):
    """
    Task 5.2: Final Integration

    Complete overview connecting all concepts with transitions.
    """

    def construct(self):
        # Title sequence
        main_title = Text("Transformer Architecture Comparison", font_size=36, color=WHITE)
        main_subtitle = Text("Encoder-Decoder vs Decoder-Only", font_size=24, color=GRAY)
        title_group = VGroup(main_title, main_subtitle).arrange(DOWN, buff=0.2)

        self.play(Write(main_title), run_time=1.5)
        self.play(FadeIn(main_subtitle))
        self.wait(1)
        self.play(FadeOut(title_group))

        # === SECTION 1: Architecture Overview ===
        section1 = Text("1. Architecture", font_size=28, color=WHITE)
        section1.to_edge(UP)
        self.play(Write(section1))

        # Mini architecture diagrams
        # Encoder-Decoder
        enc_dec_label = Text("Encoder-Decoder", font_size=16, color=BLUE)
        enc_dec_label.shift(LEFT * 3.5 + UP * 1.5)

        enc_box = VGroup(
            RoundedRectangle(width=1.5, height=1.5, color=BLUE, fill_opacity=0.2),
            Text("Encoder\n(Bi-dir)", font_size=10, color=BLUE)
        )
        enc_box[1].move_to(enc_box[0].get_center())

        dec_box_left = VGroup(
            RoundedRectangle(width=1.5, height=2.0, color=ORANGE, fill_opacity=0.2),
            Text("Decoder\n(Masked\n+Cross)", font_size=9, color=ORANGE)
        )
        dec_box_left[1].move_to(dec_box_left[0].get_center())

        enc_dec_arch = VGroup(enc_box, dec_box_left).arrange(RIGHT, buff=0.3)
        enc_dec_arch.next_to(enc_dec_label, DOWN, buff=0.2)

        cross_arrow = Arrow(
            enc_box.get_right(),
            dec_box_left.get_left(),
            color=GREEN,
            stroke_width=2,
        )

        # Decoder-Only
        dec_only_label = Text("Decoder-Only", font_size=16, color=ORANGE)
        dec_only_label.shift(RIGHT * 3.5 + UP * 1.5)

        dec_only_box = VGroup(
            RoundedRectangle(width=2.0, height=2.0, color=ORANGE, fill_opacity=0.2),
            Text("Decoder\n(Causal only)", font_size=11, color=ORANGE)
        )
        dec_only_box[1].move_to(dec_only_box[0].get_center())
        dec_only_box.next_to(dec_only_label, DOWN, buff=0.2)

        no_cross = Text("No Cross-Attn", font_size=10, color=RED)
        no_cross.next_to(dec_only_box, DOWN, buff=0.1)

        self.play(
            FadeIn(enc_dec_label), FadeIn(enc_dec_arch), Create(cross_arrow),
            FadeIn(dec_only_label), FadeIn(dec_only_box), FadeIn(no_cross),
        )
        self.wait(1.5)

        # Clear for next section
        self.play(
            FadeOut(section1), FadeOut(enc_dec_label), FadeOut(enc_dec_arch),
            FadeOut(cross_arrow), FadeOut(dec_only_label), FadeOut(dec_only_box),
            FadeOut(no_cross),
        )

        # === SECTION 2: Training ===
        section2 = Text("2. Training", font_size=28, color=WHITE)
        section2.to_edge(UP)
        self.play(Write(section2))

        # Training comparison
        training_sim = VGroup(
            Text("Similarities:", font_size=16, color=GREEN),
            Text("• Teacher Forcing", font_size=13, color=WHITE),
            Text("• Parallel computation", font_size=13, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        training_sim.shift(LEFT * 3 + UP * 0.5)

        training_diff = VGroup(
            Text("Differences:", font_size=16, color=RED),
            Text("Enc-Dec: Bidirectional encoding", font_size=13, color=BLUE),
            Text("Dec-Only: Causal mask only", font_size=13, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        training_diff.shift(RIGHT * 2.5 + UP * 0.5)

        # Causal mask mini visualization
        causal_mini = AttentionMatrix(
            size=4,
            tokens=["", "", "", ""],
            attention_type="causal",
            cell_size=0.25,
            show_labels=False,
        )
        causal_mini.scale(0.8)
        causal_mini.next_to(training_diff, DOWN, buff=0.3)

        causal_label = Text("Causal Mask", font_size=10, color=ORANGE)
        causal_label.next_to(causal_mini, DOWN, buff=0.1)

        self.play(FadeIn(training_sim), FadeIn(training_diff))
        self.play(FadeIn(causal_mini), FadeIn(causal_label))
        self.wait(1.5)

        self.play(
            FadeOut(section2), FadeOut(training_sim), FadeOut(training_diff),
            FadeOut(causal_mini), FadeOut(causal_label),
        )

        # === SECTION 3: Inference ===
        section3 = Text("3. Inference", font_size=28, color=WHITE)
        section3.to_edge(UP)
        self.play(Write(section3))

        # Inference comparison
        enc_dec_inf = VGroup(
            Text("Encoder-Decoder:", font_size=16, color=BLUE),
            Text("1. Encode once", font_size=12, color=WHITE),
            Text("2. Decode step by step", font_size=12, color=WHITE),
            Text("3. Two caches:", font_size=12, color=WHITE),
            Text("   - Encoder states", font_size=11, color=GREEN),
            Text("   - Decoder KV", font_size=11, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        enc_dec_inf.shift(LEFT * 3 + UP * 0.3)

        dec_only_inf = VGroup(
            Text("Decoder-Only:", font_size=16, color=ORANGE),
            Text("1. Prefill (parallel)", font_size=12, color=WHITE),
            Text("2. Generate (sequential)", font_size=12, color=WHITE),
            Text("3. Single KV Cache", font_size=12, color=YELLOW),
            Text("   grows with each token", font_size=11, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        dec_only_inf.shift(RIGHT * 2.5 + UP * 0.3)

        self.play(FadeIn(enc_dec_inf), FadeIn(dec_only_inf))
        self.wait(1.5)

        self.play(FadeOut(section3), FadeOut(enc_dec_inf), FadeOut(dec_only_inf))

        # === SECTION 4: Use Cases ===
        section4 = Text("4. Use Cases", font_size=28, color=WHITE)
        section4.to_edge(UP)
        self.play(Write(section4))

        enc_dec_use = VGroup(
            Text("Encoder-Decoder", font_size=18, color=BLUE),
            Text("Best for:", font_size=14, color=WHITE),
            Text("• Translation", font_size=13, color=GRAY),
            Text("• Summarization", font_size=13, color=GRAY),
            Text("• Seq2Seq tasks", font_size=13, color=GRAY),
            Text("", font_size=8),
            Text("Models: T5, BART, mBART", font_size=11, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        enc_dec_use.shift(LEFT * 3)

        dec_only_use = VGroup(
            Text("Decoder-Only", font_size=18, color=ORANGE),
            Text("Best for:", font_size=14, color=WHITE),
            Text("• Text generation", font_size=13, color=GRAY),
            Text("• Completion", font_size=13, color=GRAY),
            Text("• Chat/dialogue", font_size=13, color=GRAY),
            Text("", font_size=8),
            Text("Models: GPT, LLaMA, Mistral", font_size=11, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        dec_only_use.shift(RIGHT * 3)

        self.play(FadeIn(enc_dec_use), FadeIn(dec_only_use))
        self.wait(2)

        self.play(FadeOut(section4), FadeOut(enc_dec_use), FadeOut(dec_only_use))

        # === FINALE ===
        finale_title = Text("Key Takeaways", font_size=32, color=WHITE)
        finale_title.to_edge(UP)
        self.play(Write(finale_title))

        takeaways = VGroup(
            Text("1. Architecture: Enc-Dec has separate stacks, Dec-Only is unified", font_size=14, color=WHITE),
            Text("2. Attention: Enc-Dec uses bidirectional + cross, Dec-Only is causal only", font_size=14, color=WHITE),
            Text("3. Training: Both parallel, but different attention patterns", font_size=14, color=WHITE),
            Text("4. Inference: Different caching strategies (2 vs 1 cache)", font_size=14, color=WHITE),
            Text("5. Use case determines architecture choice", font_size=14, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        takeaways.move_to(ORIGIN)

        for i, takeaway in enumerate(takeaways):
            self.play(FadeIn(takeaway, shift=RIGHT * 0.3))
            self.wait(0.5)

        self.wait(2)

        # End
        end_text = Text("Thank you!", font_size=36, color=GREEN)
        end_text.to_edge(DOWN, buff=1)
        self.play(FadeIn(end_text))
        self.wait(2)


class TransformerComparison(Scene):
    """
    Main entry point that combines key scenes for a cohesive presentation.
    This is a shorter version suitable for testing the complete flow.
    """

    def construct(self):
        # Title
        title = Text("Encoder-Decoder vs Decoder-Only", font_size=36, color=WHITE)
        subtitle = Text("Transformer Architecture Comparison", font_size=24, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.2)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title_group))

        # Quick architecture overview
        arch_title = Text("Architecture Overview", font_size=28, color=WHITE)
        arch_title.to_edge(UP)
        self.play(Write(arch_title))

        # Side by side comparison
        left = VGroup(
            Text("Encoder-Decoder", font_size=18, color=BLUE),
            VGroup(
                RoundedRectangle(width=1.2, height=1.0, color=BLUE, fill_opacity=0.2),
                Text("Enc", font_size=12, color=BLUE),
            ),
            VGroup(
                RoundedRectangle(width=1.2, height=1.5, color=ORANGE, fill_opacity=0.2),
                Text("Dec", font_size=12, color=ORANGE),
            ),
        )
        left[1][1].move_to(left[1][0].get_center())
        left[2][1].move_to(left[2][0].get_center())
        left[1:].arrange(RIGHT, buff=0.3)
        left.arrange(DOWN, buff=0.2)

        right = VGroup(
            Text("Decoder-Only", font_size=18, color=ORANGE),
            VGroup(
                RoundedRectangle(width=1.5, height=2.0, color=ORANGE, fill_opacity=0.2),
                Text("Decoder", font_size=12, color=ORANGE),
            ),
        )
        right[1][1].move_to(right[1][0].get_center())
        right.arrange(DOWN, buff=0.2)

        left.shift(LEFT * 3)
        right.shift(RIGHT * 3)

        self.play(FadeIn(left), FadeIn(right))
        self.wait(1)

        # Key points
        points = VGroup(
            Text("Enc-Dec: Cross-Attention connects encoder to decoder", font_size=12, color=GREEN),
            Text("Dec-Only: No cross-attention, causal mask only", font_size=12, color=RED),
        ).arrange(DOWN, buff=0.1)
        points.to_edge(DOWN, buff=1)

        self.play(FadeIn(points))
        self.wait(2)

        self.play(
            FadeOut(arch_title), FadeOut(left), FadeOut(right), FadeOut(points)
        )

        # Final summary table (simplified)
        summary_title = Text("Quick Comparison", font_size=28, color=WHITE)
        summary_title.to_edge(UP)
        self.play(Write(summary_title))

        table_data = [
            ["", "Enc-Dec", "Dec-Only"],
            ["Cross-Attn", "Yes", "No"],
            ["Encoder", "Yes", "No"],
            ["KV Cache", "Two", "One"],
        ]

        # Simple table
        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1},
        ).scale(0.5)
        table.move_to(ORIGIN)

        self.play(FadeIn(table))
        self.wait(2)

        # End
        self.play(FadeOut(summary_title), FadeOut(table))

        end = Text("Complete!", font_size=36, color=GREEN)
        self.play(FadeIn(end))
        self.wait(2)
