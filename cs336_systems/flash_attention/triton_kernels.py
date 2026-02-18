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
    K_TILE_SIZE : tl.constexpr,
    is_causal : tl.constexpr
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
                                    shape=(N_queries,),
                                    strides=(stride_lq,),
                                    offsets=(query_tile_idx*Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,)
                                )
    
    tile_output = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    query_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    
    max_el = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    norm_factor = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    
    for i in range(tl.cdiv(N_keys,K_TILE_SIZE)):
        key_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        value_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        
        mat_mul = tl.dot(query_tile.to(tl.float16),tl.trans(key_tile).to(tl.float16)) * scale
        
        if is_causal:
            q_indices = tl.arange(0,Q_TILE_SIZE)[:,None] + query_tile_idx*Q_TILE_SIZE 
            k_indices = tl.arange(0,K_TILE_SIZE)[None,:] + K_TILE_SIZE*i
            causal_mask = q_indices >= k_indices
            mat_mul = tl.where(causal_mask, mat_mul, float('-inf'))
        
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
    

@triton.jit    
def flash_attention_backward(
    dO_ptr, O_ptr, Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_ob, stride_oq, stride_od,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_queries, N_keys, 
    scale,
    D : tl.constexpr,
    Q_TILE_SIZE : tl.constexpr,
    K_TILE_SIZE : tl.constexpr,
    is_causal : tl.constexpr      
    ):
    kv_tile_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    dO_block_ptr = tl.make_block_ptr(dO_ptr + batch_idx*stride_ob,
                                    shape=(N_queries,D),
                                    strides=(stride_oq,stride_od),
                                    offsets=(0,0),
                                    block_shape=(Q_TILE_SIZE,D),
                                    order=(1,0))
    
    Q_block_ptr = tl.make_block_ptr(base=Q_ptr + batch_idx*stride_qb,
                                    shape=(N_queries,D),
                                    strides=(stride_qq,stride_qd),
                                    offsets=(0,0),
                                    block_shape=(Q_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    K_block_ptr = tl.make_block_ptr(base=K_ptr + batch_idx*stride_kb,
                                    shape=(N_keys,D),
                                    strides=(stride_kk,stride_kd),
                                    offsets=(K_TILE_SIZE*kv_tile_id,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    V_block_ptr = tl.make_block_ptr(base=V_ptr + batch_idx*stride_vb,
                                    shape=(N_keys,D),
                                    strides=(stride_vk,stride_vd),
                                    offsets=(K_TILE_SIZE*kv_tile_id,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    L_block_ptr = tl.make_block_ptr(base=L_ptr + batch_idx*stride_lb,
                                    shape=(N_queries,),
                                    strides=(stride_lq,),
                                    offsets=(0,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,)
                                )
    
    D_block_ptr = tl.make_block_ptr(base=D_ptr + batch_idx*stride_db,
                                    shape=(N_queries,),
                                    strides=(stride_dq,),
                                    offsets=(0,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,)
                                    )
    
    dK_block_ptr = tl.make_block_ptr(base=dK_ptr + batch_idx*stride_kb,
                                    shape=(N_queries,D),
                                    strides=(stride_kk,stride_kd),
                                    offsets=(kv_tile_id*K_TILE_SIZE,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    
    dV_block_ptr = tl.make_block_ptr(base=dV_ptr + batch_idx*stride_vb,
                                    shape=(N_queries,D),
                                    strides=(stride_vk,stride_vd),
                                    offsets=(kv_tile_id*K_TILE_SIZE,0),
                                    block_shape=(K_TILE_SIZE,D),
                                    order=(1,0)
                                )
    

    dK_tile = tl.zeros((K_TILE_SIZE,D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE,D), dtype=tl.float32)
    
    K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")    
    K_tile_T = tl.trans(K_tile)
    V_tile_T = tl.trans(V_tile)
    
    for i in range(tl.cdiv(N_queries,Q_TILE_SIZE)):
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0,1), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        
        S_tile = tl.dot(Q_tile,K_tile_T)/scale
        P_tile = tl.exp(S_tile-L_tile[:,None])
        dP_tile = tl.dot(dO_tile.to(tl.float16), V_tile_T.to(tl.float16))
        
        D_tile = tl.load(D_block_ptr, boundary_check=(0,1), padding_option="zero")
        dS_tile = P_tile * (dP_tile - D_tile[:,None])/scale
        
        dV_tile = tl.dot(tl.trans(P_tile).to(tl.float16), dO_tile.to(tl.float16), acc=dV_tile)
        dK_tile = tl.dot(tl.trans(dS_tile).to(tl.float16), Q_tile.to(tl.float16), acc=dK_tile)
        
        dq_partial = tl.dot(dS_tile.to(tl.float16), K_tile.to(tl.float16))
        rm = i*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        rn = tl.arange(0, D)
        dq_ptrs = (dQ_ptr + batch_idx * stride_qb) + (rm[:, None] * stride_qq + rn[None, :])
        tl.atomic_add(dq_ptrs, dq_partial, mask=(rm[:, None] < N_queries))
        
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE,0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE,0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
    
    tl.store(dK_block_ptr, dK_tile.to(dK_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile.to(dV_ptr.dtype.element_ty), boundary_check=(0, 1))
    

@triton.jit
def compute_D_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_db, stride_dq,
    N_queries, D_model,
    BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D_model
    
    o_row_ptr = O_ptr + batch_idx * stride_ob + row_idx * stride_oq + cols
    do_row_ptr = dO_ptr + batch_idx * stride_ob + row_idx * stride_oq + cols
    
    o = tl.load(o_row_ptr, mask=mask, other=0.0)
    do = tl.load(do_row_ptr, mask=mask, other=0.0)
    
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=0)
    
    d_ptr = D_ptr + batch_idx * stride_db + row_idx * stride_dq
    tl.store(d_ptr, d_val)