import numpy as np
import torch

MAX_SEQ_LEN = 56

# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss

def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1,
                                   pre_group_num:int=0,
                                   mem=False):
    """
    For a single row block get random row attention.
    Args:
        block_id: int. block id of row.
        to_start_block_id: int. random attention coloum start id.
        to_end_block_id: int. random attention coloum end id.
        num_rand_blocks: int. number of random blocks to be selected.
        window_block_left: int. number of blocks of window to left of a block.
        window_block_right: int. number of blocks of window to right of a block.
        global_block_left: int. Number of blocks globally used to the left.
        global_block_right: int. Number of blocks globally used to the right.
        pre_group_num: int. Number of groups from the previous text that require attention retention.
    Returns:
        row containing the random attention vector of size num_rand_blocks.
    """
    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(to_start_block_id, to_end_block_id,
                              dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)

    # illegal blocks for the current block id, using window
    if not mem:
        illegal_blocks = list(
            range((block_id//window_block_left -pre_group_num)*window_block_right, 
                  (block_id//window_block_left+1)*window_block_right ))
    else:
        illegal_blocks = []

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id)))

    '''# The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)'''

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)

def bigbird_block_rand_mask_with_head(from_seq_length,
                                      to_seq_length,
                                      from_block_size,
                                      to_block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      pre_group_num,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1,
                                      mem=False):
    """Create adjacency list of random attention.
    
    [1]'window_block_left' & 'window_block_right' 用于限制 window block 的范围;
        选择 random attention 时不在 window block 所覆盖的范围内进行.
    [2]'global_block_left' & 'global_block_right' 用于设置整个注意力矩阵在左右两侧空出的 global block.
    
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_heads: int. total number of heads.
    plan_from_length: list. plan from lenght where num_rand are choosen from.
    plan_num_rand_blocks: list. number of rand blocks within the plan.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_top: int. number of blocks at the top.
    global_block_bottom: int. number of blocks at the bottom.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.
  Returns:
    adjacency list of size n_heads where each element is of size
    from_seq_length//from_block_size-2 by num_rand_blocks
  """
    '''assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"'''

    assert from_seq_length in plan_from_length, \
        "Error from sequence length not in plan!"
    # Total number of blocks in the mmask
    num_blocks = from_seq_length // from_block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // from_block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(from_seq_length)
    # Random Attention adjajency list
    assert max_plan_idx>=0, "ValueError max_plan_idx should be non negative."
    assert num_blocks>=0, "ValueError num_blocks should be non negative."
    rand_attn = [np.zeros((num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])),
                          dtype=np.int32) for i in range(num_heads)]
    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx + 1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                for blk_rw_idx in range(global_block_top,
                                        plan_block_length[plan_idx - 1]):
                    for h in range(num_heads):
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=plan_block_length[plan_idx - 1],
                            to_end_block_id=plan_block_length[plan_idx],
                            num_rand_blocks=plan_num_rand_blocks[plan_idx],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right,
                            pre_group_num=pre_group_num,
                            mem=mem)

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(plan_block_length[plan_idx - 1],
                                        plan_block_length[plan_idx]):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id - 1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                    for h in range(num_heads):
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=to_start_block_id,
                            to_end_block_id=plan_block_length[pl_id],
                            num_rand_blocks=plan_num_rand_blocks[pl_id],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right,
                            mem=mem)

        if plan_num_rand_blocks[plan_idx] == 0:
            continue
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx - 1]
            to_start_block_id = plan_block_length[plan_idx - 1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):
                rand_attn[h][blk_rw_idx,
                rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                    block_id=blk_rw_idx,
                    to_start_block_id=to_start_block_id,
                    to_end_block_id=plan_block_length[plan_idx],
                    num_rand_blocks=plan_num_rand_blocks[plan_idx],
                    window_block_left=window_block_left,
                    window_block_right=window_block_right,
                    global_block_left=global_block_left,
                    global_block_right=global_block_right,
                    pre_group_num=pre_group_num,
                    mem=mem)

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                                       global_block_bottom, :]
    return rand_attn

def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.
    Args:
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
        num_rand_blocks: int. Number of random chunks per row.
        Returns:
        plan_from_length: ending location of from block
        plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks // 2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks

def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.
  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
    return rand_attn

def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      num_rand_blocks,
                      window_attn_size_from,
                      window_attn_size_to,
                      pre_group_num,
                      rand_attn=None,
                      mem=False,
                      focus=1024):
    """Calculate BigBird attention pattern as a full dense matrix.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    rand_attn: adjajency matrix for random attention.
    focus: pick random mask within focus
  Returns:
    attention mask matrix of shape [from_seq_length, to_seq_length]
  """
    if rand_attn is None:
        rand_attn = bigbird_block_rand_mask(MAX_SEQ_LEN, MAX_SEQ_LEN,
                                            from_block_size, to_block_size,
                                            num_rand_blocks, focus)

    attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
    k = 0
    for i in range(0, (MAX_SEQ_LEN // from_block_size)):
        for j in rand_attn[i , :]:
            attn_mask[i * from_block_size:(i + 1) * from_block_size,
            j * to_block_size:(j + 1) * to_block_size] = 1

    if not mem:
        for k in range(MAX_SEQ_LEN // window_attn_size_from):
            k_ = k-pre_group_num
            if k_<0: k_=0
            attn_mask[(k) * window_attn_size_from:(k + 1) * window_attn_size_from,
                      (k_) * window_attn_size_to:(k + 1) * window_attn_size_to] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)

def generate_sparse_att(seq_len, n_vars, params, conj, time_start, pre_group_num=0,
                        use_mem=False, save=True,
                        num_rand_blocks=None, num_rand_blocks_mem=None,
                        num_rand_blocks_ratio=0.1):
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = seq_len
    if num_rand_blocks is not None:
        plan_num_rand_blocks_ = num_rand_blocks
    else:
        plan_num_rand_blocks_ = int(num_rand_blocks_ratio*params.seq_len)
        print('plan_num_rand_blocks_ = ', plan_num_rand_blocks_)
    rand_attn_ = bigbird_block_rand_mask_with_head(from_seq_length=params.seq_len, 
                                                   to_seq_length=params.seq_len, 
                                                   from_block_size=1, to_block_size=1, 
                                                   num_heads=conj.model.n_heads, 
                                                   plan_from_length=[params.seq_len], 
                                                   plan_num_rand_blocks=[plan_num_rand_blocks_],
                                                   pre_group_num=pre_group_num,
                                                   window_block_left=n_vars, 
                                                   window_block_right=n_vars, 
                                                   global_block_top=0, global_block_bottom=0, 
                                                   global_block_left=0, global_block_right=0, 
                                                   mem=False)
    sparse_attn_mask = [
        full_bigbird_mask(from_seq_length=params.seq_len, 
                          to_seq_length=params.seq_len,
                          from_block_size=1,
                          to_block_size=1, 
                          num_rand_blocks=20,
                          window_attn_size_from=n_vars,
                          window_attn_size_to=n_vars,
                          rand_attn=rand_attn_[i],
                          pre_group_num=pre_group_num,
                          mem=False,
                          focus=-1) 
        for i in range(conj.model.n_heads)]
    
    sparse_attn_mask = np.array(sparse_attn_mask)
    if save:
        np.save(params.root_path+'sparse_attn/sparse_attn_mask/sparse_attn_mask-'+str(params.seq_len)+'_'+time_start+'.npy', 
                sparse_attn_mask, allow_pickle=True)
    if use_mem and params.mem_len>0:
        print('Generate sparse attention mask for mem.')
        if num_rand_blocks_mem is not None:
            plan_num_rand_blocks_mem = num_rand_blocks_mem
        else:
            plan_num_rand_blocks_mem = int(num_rand_blocks_ratio*params.mem_len)
        rand_attn_mem = bigbird_block_rand_mask_with_head(from_seq_length=params.seq_len, 
                                                          to_seq_length=params.mem_len, 
                                                          from_block_size=1, to_block_size=1, 
                                                          num_heads=conj.model.n_heads, 
                                                          plan_from_length=[params.mem_len,params.seq_len], 
                                                          plan_num_rand_blocks=[plan_num_rand_blocks_mem, 0], 
                                                          pre_group_num=pre_group_num,
                                                          window_block_left=n_vars, 
                                                          window_block_right=n_vars, 
                                                          global_block_top=0, global_block_bottom=0, 
                                                          global_block_left=0, global_block_right=0, 
                                                          mem=True)
        sparse_attn_mask_mem = [
            full_bigbird_mask(from_seq_length=params.seq_len,
                              to_seq_length=params.mem_len,
                              from_block_size=1,
                              to_block_size=1,
                              num_rand_blocks=10,
                              window_attn_size_from=n_vars,
                              window_attn_size_to=n_vars,
                              rand_attn=rand_attn_mem[i],
                              pre_group_num=pre_group_num,
                              mem=True, 
                              focus=-1) 
            for i in range(conj.model.n_heads)]
    
        sparse_attn_mask_mem = np.array(sparse_attn_mask_mem)
    
        if save:
            np.save(params.root_path+'sparse_attn/sparse_attn_mask_mem/sparse_attn_mask_mem-'+str(params.seq_len)+'_'+time_start+'.npy', 
                    sparse_attn_mask_mem, allow_pickle=True)
    else:
        print('use_mem : ', use_mem)
        print('params.mem_len : ', params.mem_len)
        sparse_attn_mask_mem = None
    return sparse_attn_mask, sparse_attn_mask_mem