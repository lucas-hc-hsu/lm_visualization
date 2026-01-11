"""
Inference stage comparison scenes for Encoder-Decoder vs Decoder-Only transformers.
Phase 4: Inference Stage Comparison
"""

from manim import *
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.transformer_block import TransformerBlock
from components.attention_matrix import AttentionMatrix


class EncoderDecoderInference(Scene):
    """
    Task 4.1: Encoder-Decoder Inference Flow

    Shows the complete inference process:

    Phase A: Encode (one-time)
    Input → Encoder → Hidden States [h1, h2, h3]
                      ↓
            Cached for Cross-Attention

    Phase B: Decode (step-by-step autoregressive)
    Step 1: [<BOS>] → Decoder → [你好]
            - Self-Attention KV Cache starts building
            - Cross-Attention queries [h1,h2,h3]

    Step 2: [<BOS>, 你好] → Decoder → [世界]
            - Self-Attention uses KV Cache
            - Cross-Attention queries [h1,h2,h3] (same)

    Key: Must show Decoder also has its own KV Cache (for Self-Attention)
    """

    def construct(self):
        # Title
        title = Text("Encoder-Decoder Inference", font_size=32, color=WHITE)
        subtitle = Text("Translation: 'Hello World' → '你好 世界'", font_size=20, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.3)

        # ===== PHASE A: ENCODING =====
        phase_a = Text("Phase A: Encoding (one-time)", font_size=20, color=BLUE)
        phase_a.shift(UP * 2 + LEFT * 4)

        self.play(FadeIn(phase_a))

        # Input tokens
        input_tokens_text = ["Hello", "World"]
        input_tokens = VGroup()
        for txt in input_tokens_text:
            token = VGroup(
                RoundedRectangle(width=0.9, height=0.5, corner_radius=0.1, color=BLUE_C, fill_opacity=0.3),
                Text(txt, font_size=14, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            input_tokens.add(token)

        input_tokens.arrange(RIGHT, buff=0.1)
        input_tokens.shift(UP * 1 + LEFT * 4)

        input_label = Text("Input", font_size=12, color=GRAY)
        input_label.next_to(input_tokens, LEFT, buff=0.2)

        self.play(FadeIn(input_tokens), FadeIn(input_label))

        # Encoder block
        encoder_block = VGroup(
            RoundedRectangle(width=2.5, height=0.8, color=BLUE, fill_opacity=0.2),
            Text("Encoder", font_size=14, color=BLUE)
        )
        encoder_block[1].move_to(encoder_block[0].get_center())
        encoder_block.next_to(input_tokens, DOWN, buff=0.3)

        self.play(FadeIn(encoder_block))

        # Encoder hidden states (cached)
        encoder_states = VGroup()
        for i in range(2):
            state = VGroup(
                RoundedRectangle(width=0.7, height=0.5, corner_radius=0.05, color=GREEN, fill_opacity=0.5),
                Text(f"h{i+1}", font_size=12, color=WHITE)
            )
            state[1].move_to(state[0].get_center())
            encoder_states.add(state)

        encoder_states.arrange(RIGHT, buff=0.15)
        encoder_states.next_to(encoder_block, DOWN, buff=0.3)

        cache_label = Text("Encoder Cache (K, V)", font_size=11, color=GREEN)
        cache_label.next_to(encoder_states, DOWN, buff=0.1)

        # Cache box
        cache_box = RoundedRectangle(
            width=encoder_states.width + 0.4,
            height=encoder_states.height + 0.6,
            corner_radius=0.1,
            color=GREEN,
            stroke_width=2,
            fill_opacity=0.1,
        )
        cache_box.move_to(encoder_states.get_center())

        self.play(
            *[FadeIn(state, shift=DOWN * 0.2) for state in encoder_states],
            FadeIn(cache_label),
            Create(cache_box),
        )
        self.wait(0.5)

        one_time_note = Text("(computed once, reused for all steps)", font_size=10, color=GREEN)
        one_time_note.next_to(cache_label, DOWN, buff=0.05)
        self.play(FadeIn(one_time_note))
        self.wait(0.5)

        # ===== PHASE B: DECODING =====
        phase_b = Text("Phase B: Decoding (autoregressive)", font_size=20, color=ORANGE)
        phase_b.shift(UP * 2 + RIGHT * 3)

        self.play(FadeIn(phase_b))

        # Decoder structure
        decoder_box = VGroup(
            RoundedRectangle(width=3.0, height=2.0, color=ORANGE, fill_opacity=0.15),
        )
        decoder_box.shift(RIGHT * 3 + DOWN * 0.5)

        # Decoder internals
        self_attn = VGroup(
            RoundedRectangle(width=2.6, height=0.45, corner_radius=0.05, color=YELLOW, fill_opacity=0.3),
            Text("Masked Self-Attention", font_size=10, color=YELLOW)
        )
        self_attn[1].move_to(self_attn[0].get_center())

        cross_attn = VGroup(
            RoundedRectangle(width=2.6, height=0.45, corner_radius=0.05, color=GREEN, fill_opacity=0.3),
            Text("Cross-Attention", font_size=10, color=GREEN)
        )
        cross_attn[1].move_to(cross_attn[0].get_center())

        ffn = VGroup(
            RoundedRectangle(width=2.6, height=0.35, corner_radius=0.05, color=PURPLE, fill_opacity=0.3),
            Text("FFN", font_size=10, color=PURPLE)
        )
        ffn[1].move_to(ffn[0].get_center())

        decoder_internals = VGroup(self_attn, cross_attn, ffn).arrange(DOWN, buff=0.1)
        decoder_internals.move_to(decoder_box[0].get_center())
        decoder_box.add(decoder_internals)

        decoder_label = Text("Decoder", font_size=14, color=ORANGE)
        decoder_label.next_to(decoder_box, UP, buff=0.1)

        self.play(FadeIn(decoder_box), FadeIn(decoder_label))

        # Decoder KV Cache (for self-attention)
        decoder_kv_label = Text("Decoder KV Cache", font_size=11, color=YELLOW)
        decoder_kv_label.next_to(decoder_box, RIGHT, buff=0.2).shift(UP * 0.3)

        decoder_kv_cache = VGroup()
        decoder_kv_box = RoundedRectangle(
            width=1.2,
            height=1.0,
            corner_radius=0.1,
            color=YELLOW,
            stroke_width=2,
            fill_opacity=0.1,
        )
        decoder_kv_box.next_to(decoder_kv_label, DOWN, buff=0.1)

        self.play(FadeIn(decoder_kv_label), Create(decoder_kv_box))

        # Cross-attention connection from encoder cache
        cross_arrow = CurvedArrow(
            start_point=cache_box.get_right() + RIGHT * 0.1,
            end_point=cross_attn.get_left() + LEFT * 0.15,
            color=GREEN,
            stroke_width=2,
            angle=-TAU/8,
        )
        kv_text = Text("K, V", font_size=11, color=GREEN)
        kv_text.move_to(cross_arrow.get_center() + UP * 0.3 + LEFT * 0.3)

        self.play(Create(cross_arrow), FadeIn(kv_text))

        # ===== STEP 1 =====
        step1_label = Text("Step 1", font_size=14, color=ORANGE)
        step1_label.next_to(decoder_box, DOWN, buff=0.8)

        # Input: <BOS>
        step1_input = VGroup(
            RoundedRectangle(width=0.8, height=0.45, corner_radius=0.1, color=ORANGE, fill_opacity=0.3),
            Text("<BOS>", font_size=11, color=WHITE)
        )
        step1_input[1].move_to(step1_input[0].get_center())
        step1_input.next_to(step1_label, LEFT, buff=0.5)

        step1_output = VGroup(
            RoundedRectangle(width=0.7, height=0.45, corner_radius=0.1, color=YELLOW, fill_opacity=0.4),
            Text("你好", font_size=11, color=WHITE)
        )
        step1_output[1].move_to(step1_output[0].get_center())
        step1_output.next_to(step1_label, RIGHT, buff=0.5)

        step1_arrow = Arrow(
            step1_input.get_right() + RIGHT * 0.05,
            step1_output.get_left() + LEFT * 0.05,
            color=WHITE,
            stroke_width=1.5,
        )

        self.play(FadeIn(step1_label), FadeIn(step1_input))
        self.play(Create(step1_arrow), FadeIn(step1_output))

        # Update decoder KV cache
        kv_entry1 = VGroup(
            RoundedRectangle(width=0.3, height=0.25, color=YELLOW, fill_opacity=0.4),
            Text("k1", font_size=8, color=WHITE)
        )
        kv_entry1[1].move_to(kv_entry1[0].get_center())
        kv_entry1.move_to(decoder_kv_box.get_center() + UP * 0.25)

        self.play(FadeIn(kv_entry1, shift=LEFT * 0.3))
        self.wait(0.3)

        # ===== STEP 2 =====
        step2_label = Text("Step 2", font_size=14, color=ORANGE)
        step2_label.next_to(step1_label, DOWN, buff=0.5)

        step2_input = VGroup()
        for txt in ["<BOS>", "你好"]:
            token = VGroup(
                RoundedRectangle(width=0.7, height=0.45, corner_radius=0.1, color=ORANGE, fill_opacity=0.3),
                Text(txt, font_size=10, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            step2_input.add(token)
        step2_input.arrange(RIGHT, buff=0.05)
        step2_input.next_to(step2_label, LEFT, buff=0.3)

        step2_output = VGroup(
            RoundedRectangle(width=0.7, height=0.45, corner_radius=0.1, color=YELLOW, fill_opacity=0.4),
            Text("世界", font_size=11, color=WHITE)
        )
        step2_output[1].move_to(step2_output[0].get_center())
        step2_output.next_to(step2_label, RIGHT, buff=0.5)

        step2_arrow = Arrow(
            step2_input.get_right() + RIGHT * 0.05,
            step2_output.get_left() + LEFT * 0.05,
            color=WHITE,
            stroke_width=1.5,
        )

        # KV cache indicator for step 2
        cache_note = Text("Uses cached k1", font_size=9, color=YELLOW)
        cache_note.next_to(step2_input, UP, buff=0.1)

        self.play(FadeIn(step2_label), FadeIn(step2_input), FadeIn(cache_note))
        self.play(Create(step2_arrow), FadeIn(step2_output))

        # Update decoder KV cache with second entry
        kv_entry2 = VGroup(
            RoundedRectangle(width=0.3, height=0.25, color=YELLOW, fill_opacity=0.4),
            Text("k2", font_size=8, color=WHITE)
        )
        kv_entry2[1].move_to(kv_entry2[0].get_center())
        kv_entry2.move_to(decoder_kv_box.get_center() + DOWN * 0.1)

        self.play(FadeIn(kv_entry2, shift=LEFT * 0.3))

        # Summary
        summary = VGroup(
            Text("Key Points:", font_size=14, color=WHITE),
            Text("1. Encoder runs once, output cached for all steps", font_size=11, color=GREEN),
            Text("2. Cross-Attention always queries same encoder states", font_size=11, color=GREEN),
            Text("3. Decoder has its own KV Cache for Self-Attention", font_size=11, color=YELLOW),
            Text("4. Each step adds to Decoder KV Cache", font_size=11, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        summary.to_edge(DOWN, buff=0.2)

        self.play(FadeIn(summary))
        self.wait(3)


class DecoderOnlyInference(Scene):
    """
    Task 4.2: Decoder-Only Inference Flow

    Shows two-stage inference:

    Phase A: Prefill (parallel processing of prompt)
    [The] [cat] → Decoder (parallel) → Build complete KV Cache
                  K: [k1, k2]
                  V: [v1, v2]

    Phase B: Generation (token by token)
    Step 1: New token [?]
            - Compute Q_new
            - Attend to KV Cache [k1,k2] + [k_new]
            - Output [sat]
            - Update Cache: K=[k1,k2,k3], V=[v1,v2,v3]

    Step 2: New token [?]
            - Only compute new token's Q
            - Attend to complete Cache
            - Output [on]

    Key animations:
    1. Prefill stage parallel processing
    2. KV Cache growth
    3. New token only needs Q computation, reuses old K,V
    """

    def construct(self):
        # Title
        title = Text("Decoder-Only Inference", font_size=32, color=WHITE)
        subtitle = Text("Prompt: 'The cat' → Generation", font_size=20, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.1)
        title_group.to_edge(UP)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.3)

        # ===== PHASE A: PREFILL =====
        phase_a = Text("Phase A: Prefill (parallel)", font_size=20, color=BLUE)
        phase_a.shift(UP * 2 + LEFT * 3.5)

        self.play(FadeIn(phase_a))

        # Prompt tokens
        prompt_tokens = VGroup()
        for txt in ["The", "cat"]:
            token = VGroup(
                RoundedRectangle(width=0.75, height=0.5, corner_radius=0.1, color=BLUE_C, fill_opacity=0.3),
                Text(txt, font_size=13, color=WHITE)
            )
            token[1].move_to(token[0].get_center())
            prompt_tokens.add(token)

        prompt_tokens.arrange(RIGHT, buff=0.1)
        prompt_tokens.shift(UP * 1 + LEFT * 4)

        prompt_label = Text("Prompt", font_size=12, color=GRAY)
        prompt_label.next_to(prompt_tokens, LEFT, buff=0.2)

        self.play(FadeIn(prompt_tokens), FadeIn(prompt_label))

        # Show parallel processing
        parallel_arrows = VGroup()
        for token in prompt_tokens:
            arrow = Arrow(
                token.get_bottom() + DOWN * 0.1,
                token.get_bottom() + DOWN * 0.5,
                color=BLUE_C,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.3,
            )
            parallel_arrows.add(arrow)

        parallel_note = Text("Processed in parallel", font_size=10, color=BLUE_C)
        parallel_note.next_to(parallel_arrows, DOWN, buff=0.1)

        self.play(
            *[Create(arrow) for arrow in parallel_arrows],
            FadeIn(parallel_note),
        )

        # Decoder block for prefill
        decoder_prefill = VGroup(
            RoundedRectangle(width=2.5, height=0.8, color=ORANGE, fill_opacity=0.2),
            Text("Decoder (Causal)", font_size=12, color=ORANGE)
        )
        decoder_prefill[1].move_to(decoder_prefill[0].get_center())
        decoder_prefill.next_to(parallel_note, DOWN, buff=0.2)

        self.play(FadeIn(decoder_prefill))

        # KV Cache after prefill
        kv_cache_label = Text("KV Cache", font_size=14, color=YELLOW)
        kv_cache_label.shift(UP * 1 + RIGHT * 2)

        kv_cache_box = RoundedRectangle(
            width=2.5,
            height=1.2,
            corner_radius=0.1,
            color=YELLOW,
            stroke_width=2,
            fill_opacity=0.1,
        )
        kv_cache_box.next_to(kv_cache_label, DOWN, buff=0.15)

        # Initial KV entries from prefill
        k_row = VGroup(
            Text("K:", font_size=11, color=YELLOW),
            VGroup(*[
                RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
                for _ in range(2)
            ]).arrange(RIGHT, buff=0.08)
        ).arrange(RIGHT, buff=0.1)

        v_row = VGroup(
            Text("V:", font_size=11, color=YELLOW),
            VGroup(*[
                RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
                for _ in range(2)
            ]).arrange(RIGHT, buff=0.08)
        ).arrange(RIGHT, buff=0.1)

        kv_entries = VGroup(k_row, v_row).arrange(DOWN, buff=0.15)
        kv_entries.move_to(kv_cache_box.get_center())

        # Labels for k1, k2
        k_labels = VGroup(
            Text("k1", font_size=8, color=WHITE),
            Text("k2", font_size=8, color=WHITE),
        )
        k_labels[0].move_to(k_row[1][0].get_center())
        k_labels[1].move_to(k_row[1][1].get_center())

        v_labels = VGroup(
            Text("v1", font_size=8, color=WHITE),
            Text("v2", font_size=8, color=WHITE),
        )
        v_labels[0].move_to(v_row[1][0].get_center())
        v_labels[1].move_to(v_row[1][1].get_center())

        self.play(FadeIn(kv_cache_label), Create(kv_cache_box))
        self.play(
            FadeIn(kv_entries),
            FadeIn(k_labels),
            FadeIn(v_labels),
        )

        prefill_complete = Text("Prefill complete!", font_size=11, color=GREEN)
        prefill_complete.next_to(kv_cache_box, DOWN, buff=0.1)
        self.play(FadeIn(prefill_complete))
        self.wait(0.5)

        # ===== PHASE B: GENERATION =====
        phase_b = Text("Phase B: Generation (autoregressive)", font_size=20, color=ORANGE)
        phase_b.shift(DOWN * 0.8 + LEFT * 2)

        self.play(FadeIn(phase_b))

        # Step 1: Generate "sat"
        step1_group = VGroup()

        step1_label = Text("Step 1:", font_size=12, color=ORANGE)

        # Only compute Q for new position
        new_q = VGroup(
            RoundedRectangle(width=0.5, height=0.35, color=RED, fill_opacity=0.4),
            Text("Q_new", font_size=9, color=WHITE)
        )
        new_q[1].move_to(new_q[0].get_center())

        q_note = Text("(only new Q)", font_size=9, color=RED)

        attend_arrow = Arrow(
            ORIGIN, RIGHT * 0.8,
            color=WHITE,
            stroke_width=1.5,
        )

        attend_label = Text("Attend to", font_size=9, color=WHITE)

        cache_ref = Text("[k1,k2] + k_new", font_size=9, color=YELLOW)

        output1 = VGroup(
            RoundedRectangle(width=0.6, height=0.4, color=GREEN, fill_opacity=0.4),
            Text("sat", font_size=11, color=WHITE)
        )
        output1[1].move_to(output1[0].get_center())

        step1_row = VGroup(step1_label, new_q, q_note, attend_arrow, cache_ref, output1)
        step1_row.arrange(RIGHT, buff=0.15)
        step1_row.next_to(phase_b, DOWN, buff=0.4)

        self.play(FadeIn(step1_row))

        # Add k3, v3 to cache
        k3 = RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
        k3_label = Text("k3", font_size=8, color=WHITE)
        v3 = RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
        v3_label = Text("v3", font_size=8, color=WHITE)

        k3.next_to(k_row[1][1], RIGHT, buff=0.08)
        k3_label.move_to(k3.get_center())
        v3.next_to(v_row[1][1], RIGHT, buff=0.08)
        v3_label.move_to(v3.get_center())

        cache_update1 = Text("+k3,v3", font_size=9, color=GREEN)
        cache_update1.next_to(kv_cache_box, RIGHT, buff=0.1)

        self.play(
            FadeIn(k3), FadeIn(k3_label),
            FadeIn(v3), FadeIn(v3_label),
            FadeIn(cache_update1),
        )
        self.wait(0.3)

        # Step 2: Generate "on"
        step2_label = Text("Step 2:", font_size=12, color=ORANGE)

        new_q2 = VGroup(
            RoundedRectangle(width=0.5, height=0.35, color=RED, fill_opacity=0.4),
            Text("Q_new", font_size=9, color=WHITE)
        )
        new_q2[1].move_to(new_q2[0].get_center())

        attend_arrow2 = Arrow(
            ORIGIN, RIGHT * 0.8,
            color=WHITE,
            stroke_width=1.5,
        )

        cache_ref2 = Text("[k1,k2,k3] + k_new", font_size=9, color=YELLOW)

        output2 = VGroup(
            RoundedRectangle(width=0.6, height=0.4, color=GREEN, fill_opacity=0.4),
            Text("on", font_size=11, color=WHITE)
        )
        output2[1].move_to(output2[0].get_center())

        step2_row = VGroup(step2_label, new_q2, attend_arrow2, cache_ref2, output2)
        step2_row.arrange(RIGHT, buff=0.15)
        step2_row.next_to(step1_row, DOWN, buff=0.3)

        self.play(FadeIn(step2_row))

        # Add k4, v4 to cache
        k4 = RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
        k4_label = Text("k4", font_size=8, color=WHITE)
        v4 = RoundedRectangle(width=0.4, height=0.3, color=YELLOW, fill_opacity=0.4)
        v4_label = Text("v4", font_size=8, color=WHITE)

        k4.next_to(k3, RIGHT, buff=0.08)
        k4_label.move_to(k4.get_center())
        v4.next_to(v3, RIGHT, buff=0.08)
        v4_label.move_to(v4.get_center())

        self.play(
            FadeIn(k4), FadeIn(k4_label),
            FadeIn(v4), FadeIn(v4_label),
        )
        self.wait(0.3)

        # Summary
        summary = VGroup(
            Text("Key Points:", font_size=14, color=WHITE),
            Text("1. Prefill: process prompt in parallel, build KV Cache", font_size=11, color=BLUE),
            Text("2. Generation: only compute Q for new token", font_size=11, color=RED),
            Text("3. Reuse all cached K,V from previous tokens", font_size=11, color=YELLOW),
            Text("4. KV Cache grows with each generated token", font_size=11, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        summary.to_edge(DOWN, buff=0.15)

        self.play(FadeIn(summary))
        self.wait(3)


class InferenceEfficiencyComparison(Scene):
    """
    Task 4.3: Inference Efficiency Comparison

    Visual comparison:

    | Operation       | Encoder-Decoder              | Decoder-Only       |
    |-----------------|------------------------------|---------------------|
    | Initialization  | Encode O(n²)                 | Prefill O(L²)       |
    | Per-step gen    | Self-Attn O(m) + Cross O(n)  | Self-Attn O(L)      |
    | Memory          | Encoder states + Decoder KV  | Single KV Cache     |
    """

    def construct(self):
        # Title
        title = Text("Inference Efficiency Comparison", font_size=32, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.3)

        # Create two sides
        left_title = Text("Encoder-Decoder", font_size=20, color=BLUE)
        right_title = Text("Decoder-Only", font_size=20, color=ORANGE)

        left_title.shift(LEFT * 3.5 + UP * 2)
        right_title.shift(RIGHT * 3.5 + UP * 2)

        self.play(FadeIn(left_title), FadeIn(right_title))

        # === LEFT: Encoder-Decoder ===
        # Initialization
        enc_dec_init = VGroup(
            Text("Initialization:", font_size=14, color=WHITE),
            Text("Encode input O(n²)", font_size=12, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        enc_dec_init.next_to(left_title, DOWN, buff=0.4)

        # Visual: encoder block
        enc_block_viz = VGroup(
            RoundedRectangle(width=2.0, height=0.6, color=BLUE, fill_opacity=0.2),
            Text("Encoder", font_size=10, color=BLUE),
        )
        enc_block_viz[1].move_to(enc_block_viz[0].get_center())
        enc_block_viz.next_to(enc_dec_init, DOWN, buff=0.15)

        # Per-step
        enc_dec_step = VGroup(
            Text("Per-step:", font_size=14, color=WHITE),
            Text("Self-Attn O(m)", font_size=11, color=YELLOW),
            Text("+ Cross-Attn O(n)", font_size=11, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
        enc_dec_step.next_to(enc_block_viz, DOWN, buff=0.3)

        # Memory
        enc_dec_mem = VGroup(
            Text("Memory:", font_size=14, color=WHITE),
            Text("Encoder states", font_size=11, color=GREEN),
            Text("+ Decoder KV Cache", font_size=11, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
        enc_dec_mem.next_to(enc_dec_step, DOWN, buff=0.3)

        # Memory visualization
        mem_viz_left = VGroup(
            VGroup(
                RoundedRectangle(width=1.5, height=0.4, color=GREEN, fill_opacity=0.3),
                Text("Enc States", font_size=8, color=GREEN),
            ),
            VGroup(
                RoundedRectangle(width=1.5, height=0.4, color=YELLOW, fill_opacity=0.3),
                Text("Dec KV", font_size=8, color=YELLOW),
            ),
        )
        for viz in mem_viz_left:
            viz[1].move_to(viz[0].get_center())
        mem_viz_left.arrange(DOWN, buff=0.1)
        mem_viz_left.next_to(enc_dec_mem, DOWN, buff=0.15)

        self.play(FadeIn(enc_dec_init), FadeIn(enc_block_viz))
        self.play(FadeIn(enc_dec_step))
        self.play(FadeIn(enc_dec_mem), FadeIn(mem_viz_left))

        # === RIGHT: Decoder-Only ===
        # Initialization
        dec_only_init = VGroup(
            Text("Initialization:", font_size=14, color=WHITE),
            Text("Prefill O(L²)", font_size=12, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        dec_only_init.next_to(right_title, DOWN, buff=0.4)

        # Visual: decoder block
        dec_block_viz = VGroup(
            RoundedRectangle(width=2.0, height=0.6, color=ORANGE, fill_opacity=0.2),
            Text("Decoder", font_size=10, color=ORANGE),
        )
        dec_block_viz[1].move_to(dec_block_viz[0].get_center())
        dec_block_viz.next_to(dec_only_init, DOWN, buff=0.15)

        # Per-step
        dec_only_step = VGroup(
            Text("Per-step:", font_size=14, color=WHITE),
            Text("Self-Attn O(L)", font_size=11, color=YELLOW),
            Text("(only query new token)", font_size=10, color=GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
        dec_only_step.next_to(dec_block_viz, DOWN, buff=0.3)

        # Memory
        dec_only_mem = VGroup(
            Text("Memory:", font_size=14, color=WHITE),
            Text("Single KV Cache", font_size=11, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.03)
        dec_only_mem.next_to(dec_only_step, DOWN, buff=0.3)

        # Memory visualization
        mem_viz_right = VGroup(
            RoundedRectangle(width=1.5, height=0.6, color=YELLOW, fill_opacity=0.3),
            Text("KV Cache", font_size=10, color=YELLOW),
        )
        mem_viz_right[1].move_to(mem_viz_right[0].get_center())
        mem_viz_right.next_to(dec_only_mem, DOWN, buff=0.15)

        self.play(FadeIn(dec_only_init), FadeIn(dec_block_viz))
        self.play(FadeIn(dec_only_step))
        self.play(FadeIn(dec_only_mem), FadeIn(mem_viz_right))
        self.wait(0.5)

        # Divider line
        divider = Line(
            start=UP * 2 + ORIGIN,
            end=DOWN * 2.5 + ORIGIN,
            color=GRAY,
            stroke_width=1,
        )
        self.play(Create(divider))

        # Comparison summary at bottom
        comparison_table = VGroup(
            Text("n = input length, m = output length, L = total sequence length", font_size=10, color=GRAY),
        )
        comparison_table.to_edge(DOWN, buff=0.5)

        key_insight = VGroup(
            Text("Key Insight:", font_size=14, color=WHITE),
            Text("Enc-Dec: efficient for long inputs, short outputs", font_size=11, color=BLUE),
            Text("Dec-Only: simpler memory, grows with total length", font_size=11, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        key_insight.next_to(comparison_table, UP, buff=0.2)

        self.play(FadeIn(key_insight), FadeIn(comparison_table))
        self.wait(3)
