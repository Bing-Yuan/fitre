#!/usr/bin/env python3

import math

import torch


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    # return torch.sum(torch.stack([torch.sum(x*y) for (x, y) in zip(xs, ys)]))
    return sum([torch.sum(x*y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i]*alpha) 
    return params


def hessian_clip_cr(grads_fun, grads_data, params, delta, vs, weight=0.0):
    """ compute the optimal solution of CR subproblem along one direction vs
            min_{a > 0} a* g^Tv + a^2/2 * v^THv  + sigma/3 a^3 ||v||^3
            a = (-v^THv + sqrt((v^THv)^2 - 4\sigma ||v||^3 g^Tv))/(2*\sigma*||v||^3)
    :param grads_fun: Hessian
    :param grads_data: gradient
    :param params: model parameters
    :param delta: cubic regularization sigma = 1/delta
    :param vs: CR direction
    :param weight: H + weight* I
    :return: sub_opt CR solution and model value
    """
    sigma = 1.0/delta
    Hvs = torch.autograd.grad(grads_fun, params, grad_outputs=vs, only_inputs=True, retain_graph=True)
    Hvs = [hd.detach() + weight*d for hd, d in zip(Hvs, vs)]
    vHv = group_product(Hvs, vs)
    vnorm = math.sqrt(group_product(vs,vs))
    gv = group_product(grads_data, vs)
    alpha = ( -vHv + math.sqrt((vHv)**2 + 4*sigma*(vnorm**3)*abs(gv)))/2.0/(sigma*(vnorm**3))
    #print('alpha:', alpha.item())
    if gv > 0:
        vs = [-v*alpha for v in vs]
    else:
        vs = [v *alpha for v in vs]
    m = vHv*0.5*alpha*alpha - abs(gv)*alpha + sigma/3.0 * (alpha**3)*(vnorm**3)
    return vs, m.item()
    

def hessian_clip(grads_fun, grads_data, params, delta, vs, weight=0.0):
    """ compute the optimal solution of TR subproblem along one direction vs

    :param grads_fun: Hessian
    :param grads_data: gradient
    :param params: model parameters
    :param delta: TR radius
    :param vs: TR direction
    :param weight: H + weight* I
    :return: sub_opt TR solution and model value
    """
    Hvs = torch.autograd.grad(grads_fun, params, grad_outputs=vs, only_inputs=True, retain_graph=True)

    #for i in Hvs: 
    #    print( i.shape )

    Hvs = [hd.detach() + weight*d for hd, d in zip(Hvs, vs)]
    #print( 'Nomr of vs: ', math.sqrt(  group_product( vs, vs ) ) )
    #print( 'Norm of Hvs: ', math.sqrt( group_product( Hvs, Hvs ) ) )
    vHv = group_product(Hvs, vs)
    #print( 'vHv: ', vHv.item () )
    vnorm = math.sqrt(group_product(vs,vs))
    if vHv < 0:
        #print('NC')
        vs = [v*delta/vnorm  for v in vs]
        #print( 'Grad * vs: ', group_product(grads_data, vs).item () )
        #print( 'vHv term: ', vHv *0.5 *delta*delta/vnorm/vnorm )
        m = vHv *0.5 *delta*delta/vnorm/vnorm - group_product(grads_data, vs)
        #print('alpha:', delta/vnorm)
        return vs, m.item()
    else:
        gv = group_product(vs, grads_data)
        #print( ' grad-vec ', gv )
        alpha =  gv/(vHv + 1e-6)
        alpha = min(alpha.item(), delta/(vnorm+1e-16))
        #print('alpha:', alpha)
        # print(alpha, delta, vnorm)
        vs = [v * alpha for v in vs]
        m = vHv *0.5 *alpha*alpha - gv *alpha
        return vs, m.item()
