""" Summarized experiments using SGD. """

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from common_exp import *


class SGDArgParser(ExpArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(description='FITRE experiment -- using SGD', *args, **kwargs)
        self.set_defaults(opt='sgd')

        self.add_argument('--lr', type=float, default=0.01, metavar='LR',
                          help='learning rate (default: 0.01)')
        self.add_argument('--momentum', type=float, default=0.9, metavar='M',
                          help='SGD momentum (default: 0.9)')
        # parser.add_argument('--delta', type=float, default=1.0, metavar='D',
        #                     help='delta0 (default: 1.0)')
        # parser.add_argument('--cg', type=int, default=250, metavar='CG',
        #                     help='maximum cg iterations (default: 250)')
        # parser.add_argument('--gamma1', type=float, default=2.0, metavar='G1',
        #                     help='gamma1 (default: 2.0)')
        # parser.add_argument('--rho1', type=float, default=0.8, metavar='R1',
        #                     help='rho1 (default: 0.8)')
        # parser.add_argument('--gamma2', type=float, default=1.2, metavar='G2',
        #                     help='gamma2 (default: 1.2)')
        # parser.add_argument('--rho2', type=float, default=1e-4, metavar='R2',
        #                     help='rho2 (default: 1e-4)')
        return
    pass


def run():
    parser = SGDArgParser()
    args = parser.parse_args()
    logging.info(fmt_args(args))

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model = get_net(args).to(device)

    train_loader = get_data_loader(args, True, True)
    test_loader = get_data_loader(args, False, False)
    train_loader_org = get_data_loader(args, True, False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=0.1)

    start_epoch = -1
    save_dir = Path('models')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{args.opt}_{args.benchmark}_{args.model}.pyt'
    if args.resume > 0 and save_path.exists():
        # starting from some paused epoch and state
        logging.info(f'Loading checkpoint from {save_path}')
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'].to(device))
        optimizer.load_state_dict(checkpoint['optimizer'].to(device))
        start_epoch = args.resume

    run_time = 0.
    for epoch in range(start_epoch + 1, args.epochs):
        beg = time.time()
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            logging.debug(f'Epoch {epoch}, batch {i}, loss {loss}')

        scheduler.step()
        run_time += time.time() - beg

        train_loss, train_acc = eval_test(model, train_loader_org, criterion, device)
        test_loss, test_acc = eval_test(model, test_loader, criterion, device)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)
        logging.info(f'[Epoch {epoch} Time: {pp_time(run_time)}] -- Model saved; ' +
                     f'Train loss: {train_loss}, Train accuracy: {train_acc}; ' +
                     f'Test loss: {test_loss}, Test accuracy: {test_acc}.')

    logging.info('Finished Training')
    return


if __name__ == '__main__':
    run()
    pass
