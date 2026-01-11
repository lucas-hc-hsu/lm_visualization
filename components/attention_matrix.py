"""
AttentionMatrix component for visualizing attention patterns.
Supports: bidirectional, causal (lower triangular), and dynamic masking.
"""

from manim import *
import numpy as np


class AttentionMatrix(VGroup):
    """
    Attention matrix visualization component.

    Supports:
    - Bidirectional attention (full visibility)
    - Causal attention (lower triangular mask)
    - Dynamic mask animation
    """

    COLORS = {
        "visible": YELLOW,
        "masked": GREY_D,
        "highlight": GREEN,
        "label": WHITE,
        "grid": WHITE,
    }

    def __init__(
        self,
        size: int = 5,
        tokens: list = None,
        attention_type: str = "bidirectional",
        cell_size: float = 0.5,
        show_labels: bool = True,
        show_values: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.size = size
        self.cell_size = cell_size
        self.attention_type = attention_type
        self.show_labels = show_labels
        self.show_values = show_values

        # Default tokens if not provided
        if tokens is None:
            self.tokens = [f"t{i+1}" for i in range(size)]
        else:
            self.tokens = tokens[:size]
            # Pad if necessary
            while len(self.tokens) < size:
                self.tokens.append(f"t{len(self.tokens)+1}")

        self._build_matrix()

    def _build_matrix(self):
        """Build the attention matrix visualization."""
        # Create grid of cells
        self.cells = VGroup()
        self.cell_grid = [[None for _ in range(self.size)] for _ in range(self.size)]

        for i in range(self.size):  # row (query)
            for j in range(self.size):  # col (key)
                cell = self._create_cell(i, j)
                cell.move_to(np.array([
                    j * self.cell_size,
                    -i * self.cell_size,
                    0
                ]))
                self.cells.add(cell)
                self.cell_grid[i][j] = cell

        # Center the grid
        self.cells.move_to(ORIGIN)
        self.add(self.cells)

        # Add labels if enabled
        if self.show_labels:
            self._add_labels()

    def _create_cell(self, row: int, col: int):
        """Create a single cell in the attention matrix."""
        # Determine if cell is visible based on attention type
        is_visible = self._is_cell_visible(row, col)

        cell_color = self.COLORS["visible"] if is_visible else self.COLORS["masked"]
        fill_opacity = 0.7 if is_visible else 0.2

        cell = Square(
            side_length=self.cell_size * 0.95,
            color=cell_color,
            fill_opacity=fill_opacity,
            stroke_width=1,
        )

        # Add attention value if enabled
        if self.show_values and is_visible:
            # Placeholder value
            value = Text("●", font_size=14, color=WHITE)
            value.move_to(cell.get_center())
            return VGroup(cell, value)

        return cell

    def _is_cell_visible(self, row: int, col: int) -> bool:
        """Determine if a cell should be visible based on attention type."""
        if self.attention_type == "bidirectional":
            return True
        elif self.attention_type == "causal":
            # Lower triangular: can only attend to current and previous positions
            return col <= row
        return True

    def _add_labels(self):
        """Add row and column labels (tokens)."""
        self.row_labels = VGroup()
        self.col_labels = VGroup()

        # Get the grid bounds
        grid_left = self.cells.get_left()[0]
        grid_top = self.cells.get_top()[1]

        # Column labels (Keys) - on top
        for j, token in enumerate(self.tokens):
            label = Text(token, font_size=14, color=self.COLORS["label"])
            # Position above the j-th column
            x_pos = grid_left + (j + 0.5) * self.cell_size
            y_pos = grid_top + self.cell_size * 0.4
            label.move_to(np.array([x_pos, y_pos, 0]))
            self.col_labels.add(label)

        # Row labels (Queries) - on left
        for i, token in enumerate(self.tokens):
            label = Text(token, font_size=14, color=self.COLORS["label"])
            # Position to the left of the i-th row
            x_pos = grid_left - self.cell_size * 0.4
            y_pos = grid_top - (i + 0.5) * self.cell_size
            label.move_to(np.array([x_pos, y_pos, 0]))
            self.row_labels.add(label)

        # Add axis labels
        key_label = Text("Keys", font_size=16, color=BLUE)
        key_label.next_to(self.col_labels, UP, buff=0.15)

        query_label = Text("Queries", font_size=16, color=BLUE)
        query_label.next_to(self.row_labels, LEFT, buff=0.15)
        query_label.rotate(PI/2)

        self.add(self.row_labels, self.col_labels, key_label, query_label)
        self.key_label = key_label
        self.query_label = query_label

    def get_cell(self, row: int, col: int):
        """Get a specific cell by row and column index."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.cell_grid[row][col]
        return None

    def get_row(self, row: int):
        """Get all cells in a row."""
        if 0 <= row < self.size:
            return VGroup(*self.cell_grid[row])
        return VGroup()

    def get_col(self, col: int):
        """Get all cells in a column."""
        if 0 <= col < self.size:
            return VGroup(*[self.cell_grid[i][col] for i in range(self.size)])
        return VGroup()

    def get_visible_cells(self):
        """Get all visible cells."""
        visible = VGroup()
        for i in range(self.size):
            for j in range(self.size):
                if self._is_cell_visible(i, j):
                    visible.add(self.cell_grid[i][j])
        return visible

    def get_masked_cells(self):
        """Get all masked cells."""
        masked = VGroup()
        for i in range(self.size):
            for j in range(self.size):
                if not self._is_cell_visible(i, j):
                    masked.add(self.cell_grid[i][j])
        return masked


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
