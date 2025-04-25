def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device,
               *,
               alpha: float = 0.7,  # 软目标权重
               temperature: float = 2.0,  # 温度系数
               grad_clip: float = 1.0,  # 梯度裁剪上限 (L2‑norm)
               mse_lambda: float = 0.3,
               ):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    # term of teacher is longer
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y, label) in enumerate(distill_dataloader):

        y = y[:, -1, :]
        y = y.to(device)
        label = label[:, -1]
        label = label.to(device)
        if gap > 0:
            x_teacher = x
            x_student = x[:, gap:, :]
        else:
            x_teacher = x[:, (-1) * gap:, :]
            x_student = x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output, teacher_cls = teacher_generator(x_teacher)

        # Forward pass with student generator
        student_output, student_cls = student_generator(x_student)

        # # Calculate distillation loss (MSE between teacher and student generator's outputs)
        # soft_loss = F.mse_loss(student_output, teacher_output) * (alpha * temperature ** 2)
        # hard_loss = F.mse_loss(student_output * temperature, y) * (1 - alpha)
        # distillation_loss = soft_loss + hard_loss

        # 使用温度缩放后计算 softmax 分布
        teacher_soft = F.softmax(teacher_cls.detach() / temperature, dim=1)
        student_log_soft = F.log_softmax(student_cls / temperature, dim=1)

        # 软标签学习损失：KL 散度
        soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (alpha * temperature ** 2)

        # 硬目标损失：学生分类输出和真实标签计算交叉熵
        hard_loss = F.cross_entropy(student_cls, label.long()) * (1 - alpha)
        hard_loss += F.mse_loss(student_output * temperature, y) * (1 - alpha) * mse_lambda
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        distillation_loss.backward()

        if grad_clip is not None:
            clip_grad_norm_(student_generator.parameters(), grad_clip)

        student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed