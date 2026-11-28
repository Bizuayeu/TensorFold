//! ERNIE 5.5's contract files' own tests, no GPU: test-cuda-host runs them over the runner's modules.
test {
    _ = @import("contract_ernie5_5_head.zig");
    _ = @import("contract_ernie5_5_index.zig");
    _ = @import("contract_ernie5_5_mhc.zig");
    _ = @import("contract_ernie5_5_hca.zig");
    _ = @import("contract_ernie5_5_hca_attn.zig");
    _ = @import("contract_ernie5_5_vha.zig");
    _ = @import("contract_ernie5_5_inv_rope.zig");
    _ = @import("contract_ernie5_5_out_gate.zig");
}
