package com.avtracker.mobile.tracking.bytetrack

/** Result of [linearAssignment]: matched (row, column) pairs plus the rows and columns left over. */
class Assignment(val matches: List<Pair<Int, Int>>, val unmatchedRows: List<Int>, val unmatchedCols: List<Int>)

/**
 * boxmot's `linear_assignment` (`lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`): the minimum-cost assignment in
 * which leaving a row or a column unmatched costs `thresh / 2` each, so a pair is matched only when its cost is below
 * `thresh`. Solved on the (rows + cols)-square extended matrix with the Hungarian algorithm; the optimum is the same
 * one lapjv returns (checked against lap on random matrices).
 */
fun linearAssignment(cost: Array<DoubleArray>, rows: Int, cols: Int, thresh: Double): Assignment {
    if (rows == 0 || cols == 0) return Assignment(emptyList(), (0 until rows).toList(), (0 until cols).toList())

    val n = rows + cols
    val big = 1e9
    val e = Array(n) { DoubleArray(n) { big } }
    for (i in 0 until rows) for (j in 0 until cols) e[i][j] = cost[i][j]
    for (i in 0 until rows) e[i][cols + i] = thresh / 2
    for (j in 0 until cols) e[rows + j][j] = thresh / 2
    for (i in rows until n) for (j in cols until n) e[i][j] = 0.0

    val assignedCol = hungarian(e)

    val matches = ArrayList<Pair<Int, Int>>()
    val rowMatched = BooleanArray(rows)
    val colMatched = BooleanArray(cols)
    for (i in 0 until rows) {
        val j = assignedCol[i]
        if (j in 0 until cols && cost[i][j] < thresh + 1e-12) {
            matches += i to j
            rowMatched[i] = true
            colMatched[j] = true
        }
    }
    return Assignment(
        matches,
        (0 until rows).filter { !rowMatched[it] },
        (0 until cols).filter { !colMatched[it] }
    )
}

/** Square Hungarian algorithm (potentials + augmenting paths), O(n^3). Returns the column assigned to each row. */
private fun hungarian(a: Array<DoubleArray>): IntArray {
    val n = a.size
    val u = DoubleArray(n + 1)
    val v = DoubleArray(n + 1)
    val p = IntArray(n + 1)
    val way = IntArray(n + 1)

    for (i in 1..n) {
        p[0] = i
        var j0 = 0
        val minv = DoubleArray(n + 1) { Double.POSITIVE_INFINITY }
        val used = BooleanArray(n + 1)
        do {
            used[j0] = true
            val i0 = p[j0]
            var delta = Double.POSITIVE_INFINITY
            var j1 = 0
            for (j in 1..n) {
                if (used[j]) continue
                val cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0 }
                if (minv[j] < delta) { delta = minv[j]; j1 = j }
            }
            for (j in 0..n) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta } else minv[j] -= delta
            }
            j0 = j1
        } while (p[j0] != 0)
        do {
            val j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
        } while (j0 != 0)
    }

    val result = IntArray(n)
    for (j in 1..n) result[p[j] - 1] = j - 1
    return result
}
