import triton
import triton.language as tl 

@triton.jit
def flash_attention_forward(
    q_ptr, k_ptr, v_ptr,
    o_ptr, l_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_queries, N_keys,
    scale,
    D : tl.constexpr,
    Q_TILE_SIZE : tl.constexpr,
    K_TILE_SIZE : tl.constexpr
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(base=q_ptr + batch_idx*stride_qb,
                                    shape=(N_queries,D),
                                    strides=(stride_qq,stride_qd),
                                    offsets=(query_tile_idx*Q_TILE_SIZE,0),
                                    block_shape=(Q_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    K_block_ptr = tl.make_block_ptr(base=k_ptr + batch_idx*stride_kb,
                                    shape=(N_keys,D),
                                    strides=(stride_kk,stride_kd),
                                    offsets=(0,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    
    V_block_ptr = tl.make_block_ptr(base=v_ptr + batch_idx*stride_vb,
                                    shape=(N_keys,D),
                                    strides=(stride_vk,stride_vd),
                                    offsets=(0,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    O_block_ptr = tl.make_block_ptr(base=o_ptr + batch_idx*stride_ob,
                                    shape=(N_queries,D),
                                    strides=(stride_oq,stride_od),
                                    offsets=(query_tile_idx*Q_TILE_SIZE,0),
                                    block_shape=(Q_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    L_block_ptr = tl.make_block_ptr(base=l_ptr + batch_idx*stride_lb,
                                    shape=(Q_TILE_SIZE),
                                    strides=(stride_lq),
                                    offsets=(query_tile_idx*Q_TILE_SIZE),
                                    block_shape=(Q_TILE_SIZE),
                                    order=(1,)
                                )
    
    tile_output = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    query_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    
    max_el = tl.full((Q_TILE_SIZE), value=float("-inf"), dtype=tl.float32)
    norm_factor = tl.zeros((Q_TILE_SIZE), dtype=tl.float32)
    
    for _ in range(tl.cdiv(N_keys,K_TILE_SIZE)):
        key_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        value_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        
        mat_mul = tl.dot(query_tile.to(tl.float16),tl.trans(key_tile).to(tl.float16)) * scale
        
        row_max = tl.max(mat_mul, axis=1)
        last_max_el = max_el
        max_el = tl.maximum(last_max_el,row_max)
        
        numer = tl.exp(mat_mul-max_el[:,None])
        
        rescale_factor = tl.exp(last_max_el - max_el)
        norm_factor =  rescale_factor*norm_factor + tl.sum(numer, axis=1)
        tile_output = tile_output*rescale_factor[:,None]
        tile_output = tl.dot(numer.to(tl.float16), value_tile.to(tl.float16), acc=tile_output)
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    
    tile_output = tile_output/norm_factor[:,None]    
    log_sum_exp = max_el + tl.log(norm_factor)
    
    tl.store(O_block_ptr, tile_output, boundary_check=(0,1))
    tl.store(L_block_ptr,log_sum_exp,boundary_check=(0,))    