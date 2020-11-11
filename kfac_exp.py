""" Summarized experiments using KFAC. """

from tr_kfac_opt import KFACOptimizer
from common_exp import *
from utils import fmt_args, pp_time


class KFACArgParser(ExpArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(description='FITRE experiment -- using KFAC', *args, **kwargs)
        self.set_defaults(opt='kfac')

        self.add_argument('--weight-decay', type=float, default=0, metavar='weight',
                          help='learning rate (default: 0)')
        self.add_argument('--damp', type=float, default=0.01, metavar='damp',
                          help='damping (default: 0.01)')
        self.add_argument('--max-delta', type=float, default=100, metavar='maxdelta',
                          help='max delta (default: 100)')
        self.add_argument('--check-grad', action='store_true', default=False,
                          help='gradient')
        # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        #                     help='SGD momentum (default: 0.5)')
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
    parser = KFACArgParser()
    args = parser.parse_args()
    del args.decay_epoch  # not used in kfac experiments
    logging.info(fmt_args(args))

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    model = get_net(args).to(device)
    ''' Was using DoubleTensor, but not necessary, actually slower (speed up 2x using float)
        and better (7 epochs, 70.74% using double, 72.92% using float).
    '''
    # model.double()

    train_loader = get_data_loader(args, True, True)
    test_loader = get_data_loader(args, False, False)
    train_loader_org = get_data_loader(args, True, False)

    criterion = nn.CrossEntropyLoss()
    kfac_opt = KFACOptimizer(model=model,
                             momentum=0.0,
                             stat_decay=0.8,
                             kl_clip=1e-0,
                             damping=args.damp,
                             weight_decay=args.weight_decay,
                             check_grad=args.check_grad,
                             max_delta=args.max_delta,
                             Tf=1)

    start_epoch = -1
    save_dir = Path('models')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{args.opt}_{args.benchmark}_{args.model}.pyt'
    if args.resume > 0 and save_path.exists():
        # starting from some paused epoch and state
        logging.info(f'Loading checkpoint from {save_path}')
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'].to(device))
        kfac_opt.load_state_dict(checkpoint['optimizer'].to(device))
        start_epoch = args.resume

    run_time = 0.
    tot_batches = len(train_loader)
    for epoch in range(start_epoch + 1, args.epochs):
        beg = time.time()
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # inputs, labels = inputs.double().to(device), labels.to(device)  # must have double, otherwise abort?
            inputs, labels = inputs.to(device), labels.to(device)

            def _batch_loss():
                with torch.no_grad():
                    # It's only for computing the loss, which needs no grad.
                    model.eval()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels).item()
                model.train()
                return loss

            outputs = model(inputs)
            # probs = F.softmax(outputs, dim=1)
            kfac_opt.zero_grad()
            kfac_opt.acc_stats = True
            obj = criterion(outputs, outputs.argmax(dim=1))
            # obj = criterion(outputs, torch.max(probs, 1)[1])
            # obj = criterion(outputs, noise_outputs)
            obj.backward(retain_graph=True)
            loss = criterion(outputs, labels)
            logging.debug(f'Epoch {epoch}, batch {i} / {tot_batches}, loss {loss}')
            kfac_opt.zero_grad()

            kfac_opt.acc_stats = False
            loss.backward(create_graph=True)
            kfac_opt.step(closure=_batch_loss)

        run_time += time.time() - beg

        train_loss, train_acc = eval_test(model, train_loader_org, criterion, device)
        test_loss, test_acc = eval_test(model, test_loader, criterion, device)
        torch.save({
            'model': model.state_dict(),
            'optimizer': kfac_opt.state_dict()
        }, save_path)
        logging.info(f'[Epoch {epoch} Time: {pp_time(run_time)}] -- Model saved; ' +
                     f'Train loss: {train_loss}, Train accuracy: {train_acc}; ' +
                     f'Test loss: {test_loss}, Test accuracy: {test_acc}.')

    logging.info('Finished Training')
    return


if __name__ == '__main__':
    run()
    pass
