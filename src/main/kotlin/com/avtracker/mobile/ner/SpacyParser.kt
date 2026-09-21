package com.avtracker.mobile.ner

import kotlinx.serialization.Serializable

/** parser.json written by scripts/export_parser.py. */
@Serializable
data class ParserMeta(
    val name: String,
    val width: Int,
    val seeds: List<Int>,
    val rows: List<Int>,
    val encoderDepth: Int,
    val encoderPad: Int,
    val maxoutPieces: Int,
    val hiddenWidth: Int,
    val lowerPieces: Int,
    val numFeatures: Int,
    val rootLabel: String,
    val actions: List<NerAction>,
    val tensors: List<TensorInfo>
)

/** A dependency parse: absolute head index and label per token, and which tokens start a sentence. */
class ParseResult(val heads: IntArray, val deps: List<String>, val sentenceStart: BooleanArray)

/**
 * spaCy's `parser` component (en_core_web_sm / pt_core_news_sm): the shared `tok2vec` over six lexical attributes
 * (NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE), a greedy arc-eager transition system scored by the precomputed
 * affine layer, then `deprojectivize` and the sentence boundaries derived from the tree (`set_children_from_heads`).
 * The NER needs those boundaries: it never lets an entity run across one.
 */
class SpacyParser(private val meta: ParserMeta, weights: ByteArray) {
    private val nn = NnWeights(meta.tensors, weights)
    private val tok2vec = Tok2Vec(nn, meta.width, meta.seeds, meta.rows, meta.encoderDepth, meta.encoderPad, meta.maxoutPieces)
    private val scorer = TransitionScorer(nn, meta.numFeatures, meta.hiddenWidth, meta.lowerPieces, meta.actions.size)
    private val moves = meta.actions.map { it.move[0] } // S, D, L, R, B

    /** [keys] holds six attribute keys per token, [isSpace] flags whitespace tokens. */
    fun parse(keys: List<LongArray>, isSpace: BooleanArray): ParseResult {
        val n = keys.size
        if (n == 0) return ParseResult(IntArray(0), emptyList(), BooleanArray(0))
        scorer.prepare(tok2vec.encode(keys))

        val state = State(n)
        var steps = 0
        while (!state.isFinal) {
            if (++steps > 20 * n + 100) { state.forceFinal(); break } // cannot happen with valid moves; keeps a bad model from looping
            val valid = state.validMoves()
            val best = scorer.best(state.contextTokens()) { a -> valid[moves[a]] == true }
            if (best == -1) {
                state.forceFinal()
            } else {
                state.apply(moves[best], meta.actions[best].label)
            }
        }

        // ArcEager.set_annotations
        val head = IntArray(n)
        val dep = arrayOfNulls<String>(n)
        for (arc in state.arcs()) {
            head[arc.child] = arc.head - arc.child
            dep[arc.child] = arc.label
        }
        for (i in 0 until n) if (head[i] == 0) dep[i] = meta.rootLabel

        val doc = Doc(head, dep, isSpace)
        doc.setChildrenFromHeads()
        doc.deprojectivize()
        // The pipeline's attribute_ruler then relabels every whitespace token that has a dependency as "dep".
        val deps = doc.dep.mapIndexed { i, d -> if (isSpace[i] && !d.isNullOrEmpty()) "dep" else d ?: "" }
        return ParseResult(IntArray(n) { it + doc.head[it] }, deps, BooleanArray(n) { doc.sentStart[it] == 1 })
    }

    private class Arc(val head: Int, val child: Int, val label: String)

    /** spaCy's `StateC` for the arc-eager system, including the unshift/rebuffer mechanism of 3.x. */
    private class State(private val length: Int) {
        private val stack = ArrayList<Int>()
        private val rebuffer = ArrayList<Int>()
        private var bI = 0
        private val heads = IntArray(length) { -1 }
        private val unshiftable = BooleanArray(length)
        private val leftArcs = HashMap<Int, ArrayList<Arc>>()
        private val rightArcs = HashMap<Int, ArrayList<Arc>>()
        private val sentStarts = HashSet<Int>()

        val isFinal: Boolean get() = stack.size <= 0 && bufferLength() == 0

        fun bufferLength() = (length - bI) + rebuffer.size

        fun s(i: Int): Int = if (i < 0 || i >= stack.size) -1 else stack[stack.size - (i + 1)]

        fun b(i: Int): Int {
            if (i < 0) return -1
            if (i < rebuffer.size) return rebuffer[rebuffer.size - (i + 1)]
            val index = bI + (i - rebuffer.size)
            return if (index >= length) -1 else index
        }

        private fun hasHead(child: Int) = child in 0 until length && heads[child] >= 0

        private fun isSentStart(word: Int) = word in 0 until length && word in sentStarts

        /** `nth_child`: the idx-th most recently added arc of [head] (1 = latest). */
        private fun nthChild(arcs: HashMap<Int, ArrayList<Arc>>, head: Int, idx: Int): Int {
            if (idx < 1) return -1
            val list = arcs[head] ?: return -1
            var count = 0
            for (k in list.indices.reversed()) {
                val arc = list[k]
                if (arc.child != -1) {
                    count++
                    if (count == idx) return arc.child
                }
            }
            return -1
        }

        private fun l(head: Int, idx: Int) = nthChild(leftArcs, head, idx)
        private fun r(head: Int, idx: Int) = nthChild(rightArcs, head, idx)

        /** `set_context_tokens` for the 8 features of the parser. */
        fun contextTokens(): IntArray {
            val b0 = b(0)
            val s0 = s(0)
            val ids = intArrayOf(b0, b(1), s0, s(1), s(2), l(b0, 1), l(s0, 1), r(s0, 1))
            for (i in ids.indices) if (ids[i] < 0) ids[i] = -1
            return ids
        }

        fun validMoves(): Map<Char, Boolean> {
            val shift = when {
                stack.size == 0 -> true
                bufferLength() < 2 -> false
                isSentStart(b(0)) -> false
                unshiftable.getOrElse(b(0)) { false } -> false
                else -> true
            }
            val reduce = when {
                stack.size == 0 -> false
                bufferLength() == 0 -> true
                else -> true // a token never has sent_start == -1 before parsing, so `cannot_sent_start` is false
            }
            val arc = stack.size != 0 && bufferLength() != 0 && !isSentStart(b(0))
            val brk = bufferLength() >= 2 && b(1) == b(0) + 1 && !isSentStart(b(1))
            return mapOf('S' to shift, 'D' to reduce, 'L' to arc, 'R' to arc, 'B' to brk)
        }

        fun apply(move: Char, label: String) {
            when (move) {
                'S' -> push()
                'D' -> if (hasHead(s(0)) || stack.size == 1) pop() else unshift()
                'L' -> {
                    addArc(b(0), s(0), label)
                    if (b(0) in 0 until length) unshiftable[b(0)] = false // set_reshiftable
                    pop()
                }
                'R' -> {
                    addArc(s(0), b(0), label)
                    push()
                }
                'B' -> sentStarts += b(1)
            }
        }

        private fun push() {
            val b0 = if (rebuffer.isNotEmpty()) rebuffer.removeAt(rebuffer.size - 1) else bI++
            stack += b0
        }

        private fun pop() { stack.removeAt(stack.size - 1) }

        private fun unshift() {
            val s0 = stack.removeAt(stack.size - 1)
            unshiftable[s0] = true
            rebuffer += s0
        }

        fun forceFinal() {
            stack.clear()
            rebuffer.clear()
            bI = length
        }

        private fun addArc(head: Int, child: Int, label: String) {
            if (hasHead(child)) delArc(heads[child], child)
            val arc = Arc(head, child, label)
            (if (head > child) leftArcs else rightArcs).getOrPut(head) { ArrayList() } += arc
            heads[child] = head
        }

        /**
         * `del_arc`: spaCy only removes an arc that is the last one of its head. (Its loop for the others edits a copy
         * of the arc, so they stay in place; reproduced.)
         */
        private fun delArc(head: Int, child: Int) {
            val arcs = (if (head > child) leftArcs else rightArcs)[head] ?: return
            if (arcs.isEmpty()) return
            val last = arcs.last()
            if (last.head == head && last.child == child) arcs.removeAt(arcs.size - 1)
        }

        /** Left arcs then right arcs, heads in ascending order. */
        fun arcs(): List<Arc> =
            leftArcs.keys.sorted().flatMap { leftArcs.getValue(it) } + rightArcs.keys.sorted().flatMap { rightArcs.getValue(it) }
    }

    /** The token array of a `Doc` as far as the parse post-processing needs it. */
    private class Doc(val head: IntArray, val dep: Array<String?>, private val isSpace: BooleanArray) {
        val n = head.size
        val lEdge = IntArray(n)
        val rEdge = IntArray(n)
        val sentStart = IntArray(n)

        /** `set_children_from_heads`: edges of every subtree (repeated for non-projective trees) and the sentence starts. */
        fun setChildrenFromHeads() {
            for (i in 0 until n) { lEdge[i] = i; rEdge[i] = i }
            var loop = 0
            var headsWithinSents = false
            while (!headsWithinSents) {
                headsWithinSents = setEdges()
                if (loop > 10) break
                loop++
            }
            java.util.Arrays.fill(sentStart, -1)
            for (i in 0 until n) if (head[i] == 0 && dep[i] != null) sentStart[lEdge[i]] = 1
        }

        private fun setEdges(): Boolean {
            for (i in 0 until n) {
                val h = i + head[i]
                if (lEdge[i] < lEdge[h]) lEdge[h] = lEdge[i]
                if (rEdge[i] > rEdge[h]) rEdge[h] = rEdge[i]
            }
            for (i in n - 1 downTo 0) {
                val h = i + head[i]
                if (rEdge[i] > rEdge[h]) rEdge[h] = rEdge[i]
                if (lEdge[i] < lEdge[h]) lEdge[h] = lEdge[i]
            }
            val starts = HashSet<Int>()
            for (i in 0 until n) if (head[i] == 0) starts += lEdge[i]
            var currentStart = 0
            for (i in 0 until n) {
                if ((i > 0 && i in starts) || i == n - 1) {
                    val currentEnd = i
                    for (j in currentStart until currentEnd) {
                        if (head[j] + j < currentStart || head[j] + j >= currentEnd + 1) return false
                    }
                    currentStart = i
                }
            }
            return true
        }

        /** Children as `Token.children` yields them: lefts then rights, found through the (possibly stale) edges. */
        private fun children(q: Int): List<Int> {
            val out = ArrayList<Int>()
            for (j in lEdge[q] until q) if (j + head[j] == q) out += j
            for (j in q + 1..rEdge[q]) if (j + head[j] == q) out += j
            return out
        }

        private fun findNewHead(token: Int, headLabel: String): Int {
            val tokenHead = token + head[token]
            var queue = listOf(tokenHead)
            while (queue.isNotEmpty()) {
                val next = ArrayList<Int>()
                for (q in queue) {
                    for (child in children(q)) {
                        if (isSpace[child]) continue
                        if (child == token) continue
                        if (dep[child] == headLabel) return child
                        next += child
                    }
                }
                queue = next
            }
            return tokenHead
        }

        /** `nonproj.deprojectivize`: reattach arcs whose label is decorated "label||headLabel". */
        fun deprojectivize() {
            for (i in 0 until n) {
                val label = dep[i] ?: continue
                if (DELIMITER in label) {
                    val parts = label.split(DELIMITER)
                    val newHead = findNewHead(i, parts[1])
                    head[i] = newHead - i
                    dep[i] = parts[0]
                }
            }
            setChildrenFromHeads()
        }

        companion object { const val DELIMITER = "||" }
    }
}
