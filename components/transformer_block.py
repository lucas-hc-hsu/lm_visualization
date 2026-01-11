"""
TransformerBlock component for visualizing different types of transformer blocks.
Supports: encoder, enc_dec_decoder, decoder_only
"""

from manim import *


class TransformerBlock(VGroup):
    """
    A reusable Transformer block visualization component.

    Types:
    - encoder: Bidirectional Self-Attention + FFN
    - enc_dec_decoder: Masked Self-Attention + Cross-Attention + FFN
    - decoder_only: Causal Self-Attention + FFN (no Cross-Attention)
    """

    # Color scheme
    COLORS = {
        "encoder": BLUE,
        "decoder": ORANGE,
        "attention": YELLOW,
        "cross_attention": GREEN,
        "ffn": PURPLE,
        "norm": GRAY,
        "residual": WHITE,
    }

    def __init__(
        self,
        block_type: str = "encoder",
        width: float = 3.5,
        height: float = None,
        show_residual: bool = True,
        label: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.block_type = block_type
        self._block_width = width  # Use different name to avoid conflict with VGroup.width
        self.show_residual = show_residual

        # Calculate height based on block type - increased for visibility
        if height is None:
            if block_type == "enc_dec_decoder":
                self._block_height = 5.5  # Taller for cross-attention
            else:
                self._block_height = 4.0  # Increased from 3.0
        else:
            self._block_height = height

        self._build_block(label)

    def _build_block(self, label: str = None):
        """Build the transformer block based on type."""

        # Main container
        main_color = self.COLORS["encoder"] if self.block_type == "encoder" else self.COLORS["decoder"]

        container = RoundedRectangle(
            width=self._block_width,
            height=self._block_height,
            corner_radius=0.1,
            color=main_color,
            stroke_width=2,
        )
        self.add(container)
        self.container = container

        # Build internal components based on type
        if self.block_type == "encoder":
            self._build_encoder_block()
        elif self.block_type == "enc_dec_decoder":
            self._build_enc_dec_decoder_block()
        elif self.block_type == "decoder_only":
            self._build_decoder_only_block()

        # Add label if provided
        if label:
            label_text = Text(label, font_size=20, color=main_color)
            label_text.next_to(container, UP, buff=0.1)
            self.add(label_text)
            self.label = label_text

    def _create_sublayer(self, name: str, color, width: float = None, height: float = 0.6):
        """Create a sublayer rectangle with label."""
        w = width if width else self._block_width * 0.85
        rect = RoundedRectangle(
            width=w,
            height=height,
            corner_radius=0.08,
            color=color,
            fill_opacity=0.5,  # Increased from 0.3 for better visibility
            stroke_width=2.5,  # Increased from 1.5
        )
        text = Text(name, font_size=18, color=WHITE)  # Increased font, white for contrast
        text.move_to(rect.get_center())
        group = VGroup(rect, text)
        return group

    def _create_norm_layer(self, width: float = None):
        """Create a Layer Normalization indicator."""
        w = width if width else self._block_width * 0.85
        rect = Rectangle(
            width=w,
            height=0.35,  # Increased from 0.25
            color=self.COLORS["norm"],
            fill_opacity=0.4,  # Increased from 0.2
            stroke_width=1.5,  # Increased from 1
        )
        text = Text("LayerNorm", font_size=14, color=WHITE)  # Increased font, white color
        text.move_to(rect.get_center())
        return VGroup(rect, text)

    def _create_residual_arrow(self, start, end):
        """Create a residual connection arrow."""
        # Create a curved arrow for residual connection
        path = CurvedArrow(
            start_point=start,
            end_point=end,
            color=self.COLORS["residual"],
            stroke_width=1.5,
            angle=-TAU/4,
        )
        return path

    def _build_encoder_block(self):
        """Build encoder block: Self-Attention + FFN with residual connections."""
        components = VGroup()

        # Self-Attention (Bidirectional)
        self_attn = self._create_sublayer("Self-Attention\n(Bidirectional)", self.COLORS["attention"])

        # Layer Norm after attention
        norm1 = self._create_norm_layer()

        # FFN
        ffn = self._create_sublayer("Feed Forward", self.COLORS["ffn"])

        # Layer Norm after FFN
        norm2 = self._create_norm_layer()

        # Arrange vertically
        components.add(norm2, ffn, norm1, self_attn)
        components.arrange(DOWN, buff=0.15)
        components.move_to(self.container.get_center())

        self.add(components)
        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2

        # Add residual connections if enabled
        if self.show_residual:
            self._add_residual_indicators()

    def _build_enc_dec_decoder_block(self):
        """Build encoder-decoder's decoder block: Masked Self-Attention + Cross-Attention + FFN."""
        components = VGroup()

        # Masked Self-Attention
        self_attn = self._create_sublayer("Masked Self-Attention", self.COLORS["attention"])

        # Layer Norm
        norm1 = self._create_norm_layer()

        # Cross-Attention
        cross_attn = self._create_sublayer("Cross-Attention", self.COLORS["cross_attention"])

        # Layer Norm
        norm2 = self._create_norm_layer()

        # FFN
        ffn = self._create_sublayer("Feed Forward", self.COLORS["ffn"])

        # Layer Norm
        norm3 = self._create_norm_layer()

        # Arrange vertically
        components.add(norm3, ffn, norm2, cross_attn, norm1, self_attn)
        components.arrange(DOWN, buff=0.1)
        components.move_to(self.container.get_center())

        self.add(components)
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2
        self.norm3 = norm3

        # Add K,V input indicator for cross-attention
        kv_label = Text("K,V from\nEncoder", font_size=12, color=self.COLORS["cross_attention"])
        kv_arrow = Arrow(
            start=cross_attn.get_left() + LEFT * 0.8,
            end=cross_attn.get_left() + LEFT * 0.1,
            color=self.COLORS["cross_attention"],
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2,
        )
        kv_label.next_to(kv_arrow, LEFT, buff=0.1)
        self.add(kv_arrow, kv_label)
        self.kv_arrow = kv_arrow
        self.kv_label = kv_label

    def _build_decoder_only_block(self):
        """Build decoder-only block: Causal Self-Attention + FFN (no Cross-Attention)."""
        components = VGroup()

        # Causal Self-Attention
        self_attn = self._create_sublayer("Causal Self-Attention", self.COLORS["attention"])

        # Layer Norm
        norm1 = self._create_norm_layer()

        # FFN
        ffn = self._create_sublayer("Feed Forward", self.COLORS["ffn"])

        # Layer Norm
        norm2 = self._create_norm_layer()

        # Arrange vertically
        components.add(norm2, ffn, norm1, self_attn)
        components.arrange(DOWN, buff=0.15)
        components.move_to(self.container.get_center())

        self.add(components)
        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1
        self.norm2 = norm2

        # Add residual connections if enabled
        if self.show_residual:
            self._add_residual_indicators()

    def _add_residual_indicators(self):
        """Add '+' symbols to indicate residual connections."""
        # Add small '+' symbols on the right side
        plus1 = Text("+", font_size=20, color=self.COLORS["residual"])
        plus1.next_to(self.norm1, RIGHT, buff=0.05)

        plus2 = Text("+", font_size=20, color=self.COLORS["residual"])
        plus2.next_to(self.norm2, RIGHT, buff=0.05)

        self.add(plus1, plus2)
        self.residual_indicators = VGroup(plus1, plus2)


class TestTransformerBlocks(Scene):
    """Test scene to render all three transformer block types."""

    def construct(self):
        # Title
        title = Text("Transformer Block Types", font_size=36)
        title.to_edge(UP)
        self.add(title)

        # Create three block types
        encoder_block = TransformerBlock(
            block_type="encoder",
            label="Encoder Block"
        )

        enc_dec_decoder_block = TransformerBlock(
            block_type="enc_dec_decoder",
            label="Enc-Dec Decoder Block"
        )

        decoder_only_block = TransformerBlock(
            block_type="decoder_only",
            label="Decoder-Only Block"
        )

        # Arrange horizontally
        blocks = VGroup(encoder_block, enc_dec_decoder_block, decoder_only_block)
        blocks.arrange(RIGHT, buff=1.0)
        blocks.move_to(ORIGIN)

        # Animate
        self.play(FadeIn(encoder_block))
        self.wait(0.5)
        self.play(FadeIn(enc_dec_decoder_block))
        self.wait(0.5)
        self.play(FadeIn(decoder_only_block))
        self.wait(2)
