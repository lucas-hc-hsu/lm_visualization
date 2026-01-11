"""Test file for AttentionMatrix component."""

from manim import *
from components.attention_matrix import AttentionMatrix


class TestAttentionMatrix(Scene):
    """Test scene to render attention matrices."""

    def construct(self):
        # Title
        title = Text("Attention Matrix Visualization", font_size=32)
        title.to_edge(UP)
        self.add(title)

        # Create bidirectional attention matrix
        tokens = ["The", "cat", "sat", "on", "mat"]

        bi_matrix = AttentionMatrix(
            size=5,
            tokens=tokens,
            attention_type="bidirectional",
            cell_size=0.5,
        )
        bi_label = Text("Bidirectional\n(Encoder)", font_size=18, color=BLUE)

        # Create causal attention matrix
        causal_matrix = AttentionMatrix(
            size=5,
            tokens=tokens,
            attention_type="causal",
            cell_size=0.5,
        )
        causal_label = Text("Causal\n(Decoder)", font_size=18, color=ORANGE)

        # Position matrices
        bi_matrix.shift(LEFT * 3.5)
        bi_label.next_to(bi_matrix, DOWN, buff=0.3)

        causal_matrix.shift(RIGHT * 3.5)
        causal_label.next_to(causal_matrix, DOWN, buff=0.3)

        # Animate bidirectional matrix
        self.play(FadeIn(bi_matrix), FadeIn(bi_label))
        self.wait(1)

        # Animate causal matrix
        self.play(FadeIn(causal_matrix), FadeIn(causal_label))
        self.wait(1)

        # Highlight the difference - show masking effect
        explanation = Text(
            "Causal mask: each position can only attend to itself and earlier positions",
            font_size=16,
            color=YELLOW
        )
        explanation.to_edge(DOWN)
        self.play(FadeIn(explanation))

        # Highlight masked cells
        masked_cells = causal_matrix.get_masked_cells()
        self.play(
            masked_cells.animate.set_fill(RED, opacity=0.5),
            run_time=1
        )
        self.wait(2)


class CausalMaskBuildAnimation(Scene):
    """Animation showing the causal mask being built step by step."""

    def construct(self):
        title = Text("Building Causal Mask", font_size=32)
        title.to_edge(UP)
        self.add(title)

        tokens = ["The", "cat", "sat", "on", "mat"]

        # Start with empty matrix (all masked)
        matrix = AttentionMatrix(
            size=5,
            tokens=tokens,
            attention_type="bidirectional",
            cell_size=0.6,
        )

        # Initially set all cells to masked appearance
        for i in range(5):
            for j in range(5):
                cell = matrix.get_cell(i, j)
                cell.set_fill(GREY_D, opacity=0.2)
                cell.set_stroke(GREY_D)

        self.add(matrix)
        self.wait(0.5)

        # Animate building the causal mask row by row
        for i in range(5):
            row_cells = []
            for j in range(i + 1):  # Only up to and including diagonal
                cell = matrix.get_cell(i, j)
                row_cells.append(cell)

            # Highlight current row's visible cells
            self.play(
                *[cell.animate.set_fill(YELLOW, opacity=0.7).set_stroke(YELLOW)
                  for cell in row_cells],
                run_time=0.5
            )

            # Show which token is attending
            attending_text = Text(
                f'"{tokens[i]}" can see: {", ".join(tokens[:i+1])}',
                font_size=18,
                color=GREEN
            )
            attending_text.to_edge(DOWN)

            if i == 0:
                self.play(FadeIn(attending_text))
            else:
                self.play(Transform(attending_text, attending_text))

            self.wait(0.3)

        self.wait(2)
