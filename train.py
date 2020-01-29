def train_Qnet(Q_model, model, epoch = 1000, dt = 0.01):
    optimizer = torch.optim.Adam(Q_model.parameters(), lr=0.01)
    for ep in range(epoch):
        x = X_train[0].double()
        Q =[x[0]]
        loss = 0
        for i in range(0, len(x) - 1):
            q = Q_model(x[i])  #predict next location
            #loss += L2_loss(x[i + 1], q) + Euler_Lagrange(model, Q[-1], q, dt)
            #loss += Euler_Lagrange(model, Q[-1], q, dt)
            loss += variation_integrate(Q_model, model, x[i], x[i + 1], 10, dt)
            #print(autograd.grad(model(q), q))
            Q.append(q)
        initial_condition_loss = L2_loss(Q[-1], x[-1]) + L2_loss(Q[1], x[1])
        loss += initial_condition_loss
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if ep % 20 == 0:
            print(loss.item() - initial_condition_loss.item(), initial_condition_loss.item())
            print(loss.item())
            
            
def train_Qnet_baseline(Q_model, model, epoch = 1000, dt = 0.01):
    optimizer = torch.optim.Adam(Q_model.parameters(), lr=0.01)
    for ep in range(epoch):
        x = X_train[0].double()
        Q =[x[0]]
        loss = 0
        for i in range(0, len(x) - 1):
            q = Q_model(x[i])
            #loss += L2_loss(x[i + 1], q) + Euler_Lagrange(model, Q[-1], q, dt)
            #loss += Euler_Lagrange(model, Q[-1], q, dt)
            loss += L2_loss(x[i + 1], q)
            #print(autograd.grad(model(q), q))
            Q.append(q)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if ep % 50 == 0:
            print(loss.item())
                


def train(model, epoch = 10, dt = 0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(epoch):
        x = X_train[0].double()
        c = X_train[1].double()
        loss = 0
        for i in range(0, len(x) - 1):
            #print(x[i].double(), x[i + 1].double())
            control_loss = (Euler_Lagrange(model, x[i], x[i + 1], dt) + 1e-10) / (Euler_Lagrange(model, c[i], c[i + 1], dt) + 1e-10)
            loss += f(Euler_Lagrange_normalizer(model, x[i], x[i + 1], dt)) * f(Euler_Lagrange(model, x[i], x[i + 1], dt)) * control_loss
        if ep % 50 == 0:
            print(loss)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()