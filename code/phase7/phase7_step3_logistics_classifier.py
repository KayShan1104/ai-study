"""
Phase 7 Step 3: 物流文本分类 — 从零训练一个 BERT 分类器

目标：
1. 构建物流场景的文本分类数据集（8 类，520 条）
2. 用 BERT-base-chinese 作为 backbone，搭建分类模型
3. 完整训练 pipeline：数据 → Tokenize → 训练 → 验证 → 保存
4. 评估：classification report + 混淆矩阵
"""

import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure output directory exists
os.makedirs("code/phase7/models", exist_ok=True)

# ============================================================
# Part 1: 数据集构建
# ============================================================

print("=" * 60)
print("Part 1: 构建物流文本分类数据集")
print("=" * 60)

LABELS = [
    "shipping_delay",      # 0: 运输延误
    "customs_clearance",   # 1: 清关问题
    "tracking_issue",      # 2: 物流追踪异常
    "billing_dispute",     # 3: 账单争议
    "schedule_inquiry",    # 4: 船期/航班查询
    "cargo_damage",        # 5: 货损理赔
    "booking_request",     # 6: 订舱请求
    "general",             # 7: 其他/通用
]

# 预定义的物流场景数据集（每类 65 条，共 520 条）
DATA_SEEDS = {
    "shipping_delay": [
        "我的包裹已经晚了一周了，到底什么情况？",
        "预计的送达时间已经过了三天，怎么还没有到？",
        "订单号123456显示三天前就发货了但一直没更新",
        "这批货的ETA已经推迟了两次，客户在催",
        "我的集装箱在洋山港等了五天还没装上船",
        "船期延误了一周，我的货什么时候能到？",
        "快递显示已发货但物流信息停滞了72小时",
        "货物本该昨天到达，现在连影子都没有",
        "我的shipment被delay了，能查原因吗",
        "从宁波发出来的货，预计三天到，已经一周了",
        "航班取消了，我的空运货物怎么办？",
        "海运船晚到两天，码头已经安排不了了",
        "包裹的预计送达日期一直在往后延",
        "你们的物流太慢了，比预计晚了一个星期",
        "这票货已经卡在港口两周了，能帮忙查下吗",
        "货物在转运中心停留超过48小时没有动静",
        "我的柜子在盐田港被延误了，能加急处理吗",
        "马士基的船晚点了，我的货期受影响",
        "货物预计周一到，但现在已经周四了还没消息",
        "从上海到洛杉矶的船期延误了多久？",
        "我的包裹物流信息停在已揽收已经四天",
        "铁路运输延误，货物在郑州站滞留",
        "货柜没有赶上预定的船次，下一班是什么时候",
        "清关延迟导致我的货物晚了半个月",
        "海运时间从25天变成了40天，为什么？",
        "航班延误导致我的紧急货物无法按时送达",
        "我的集装箱在釜山港转船时被延误",
        "货物到达目的港后一直没有派送",
        "预计的交货时间推迟了，能确认新的ETA吗",
        "我的订单显示已送达但实际没有收到",
        "货物在新加坡中转停留了一周没有后续",
        "铁路货运延误，客户要求赔偿",
        "我的包裹在海关检查后被卡住",
        "船期变更通知收到，但我的货已经上船了",
        "物流跟踪显示货物还在出发港没有动",
        "从青岛发出的货预计5天，现在已经10天了",
        "货物在目的港滞留，没人联系我提货",
        "我的快递周末应该到的，现在周一了还没到",
        "货物运输途中被临时改道，延误了三天",
        "码头工人罢工导致我的集装箱无法卸载",
        "货物到达中转站后一直没有人派送",
        "我的shipment状态一直显示in transit但没更新",
        "因为台风影响，船期推迟了五天",
        "货物预计今天到达但物流还没有派送信息",
        "我的集装箱没有赶上衔接的船",
        "货物从发出到现在已经两周了还没到",
        "海运的预计到达时间一直在变化",
        "我的包裹在路上丢了似的，没有任何更新",
        "航班取消后重新安排要等一周",
        "货物到达目的港后清关用了十天还没完成",
        "我的快递说昨天到但今天也没收到",
        "货物运输时间比预计长了一倍",
        "我的货柜在洛杉矶港排队等泊位已经三天",
        "从广州发出的货，现在卡在什么地方了？",
        "船舶故障导致航线变更，我的货延误了",
        "货物到达当地配送中心后一直没有派送",
        "预计送达时间改了三次，到底哪天能到？",
        "我的包裹在航空运输中延误了",
        "货物在目的国海关被扣留了多久？",
        "海运集装箱到了但码头说还没收到通知",
        "我的货物在中转港被遗忘了吗，一周没动",
        "铁路货运因为线路检修延误了两天",
        "我的订单状态异常，物流信息不更新",
        "货物到达目的港后没有人安排卸货",
        "包裹的跟踪信息显示异常但没说是什么问题",
    ],
    "customs_clearance": [
        "货物卡在海关了，怎么加快清关速度？",
        "清关需要的文件我已经提交了，为什么还没放行？",
        "我的货被海关扣了，说要补充商业发票",
        "进口报关单填错了，能修改吗？",
        "海关要求提供原产地证明，这个怎么弄？",
        "货物在海关查验，预计要多久？",
        "我的集装箱在海关等待放行已经五天了",
        "清关代理说需要额外的装箱单",
        "海关对货物的HS编码有疑问",
        "出口报关怎么办理？需要什么材料？",
        "我的货在浦东机场海关被抽查了",
        "清关费用比预期高了很多，能解释一下吗",
        "海关说我的货物需要检验检疫证明",
        "进口许可证过期了怎么办？",
        "清关状态显示申报中，什么意思？",
        "货物被海关认定为敏感商品，怎么处理？",
        "我需要帮忙联系清关代理加急处理",
        "海关要求提供产品的成分表",
        "我的货在目的港清关时被要求缴税",
        "报关单上的金额和实际不符，有影响吗？",
        "清关完成后货物多久能从港口提出？",
        "海关说我的包裹需要缴纳关税",
        "进口货物需要3C认证吗？",
        "我的货物因为标签不合规被海关退回",
        "清关代理失联了，我的货还在海关",
        "货物在海关开箱查验后多久能恢复？",
        "我需要了解目的国的进口清关要求",
        "海关对货物价值的评估比申报高",
        "我的货被海关要求提供MSDS安全数据表",
        "清关延误导致我无法按时交货给客户",
        "海关说需要补充合同和发票原件",
        "进口化妆品的清关流程是怎样的？",
        "我的货物在海关排队等查验",
        "关税计算方式我不太明白，能解释吗？",
        "海关查验发现货物数量跟申报不符",
        "清关后货物需要转关到另一个口岸",
        "我的包裹被海关退回了，为什么？",
        "电子口岸显示清关已完成但货还没放行",
        "海关要求对货物进行品牌授权验证",
        "我需要了解跨境电商的清关政策",
        "货物在海关被查验后发现包装破损",
        "清关费用包含哪些项目？",
        "我的货需要办理进出口权才能清关吗？",
        "海关说我的产品需要CCC强制认证",
        "进口食品清关需要什么特殊文件？",
        "货物在海关等待缴税通知",
        "我的报关行说需要提单原件才能清关",
        "海关查验率最近提高了吗？",
        "我的货物在保税区清关后怎么转到国内？",
        "海关要求提供产品的中文标签",
        "清关代理说我的HS编码归类有误",
        "货物因为缺少进口许可证被海关扣留",
        "我需要帮忙追踪清关进度",
        "海关查验后货物有损失谁来负责？",
        "电子报关系统提交后多久能审核通过？",
        "我的货在海关需要补充什么文件？",
        "货物被海关认定为侵权产品怎么办？",
        "清关完成后怎么查询货物放行状态？",
        "海关对危险品有什么特殊清关要求？",
        "我的货物在海关需要支付哪些费用？",
        "进口医疗器械清关需要什么资质？",
        "海关说我的货物需要熏蒸证明",
        "货物清关后从哪个港口提货？",
        "我的包裹在海关显示待缴税怎么处理？",
        "清关代理说需要补充原产地证书",
    ],
    "tracking_issue": [
        "单号查不到物流信息，是不是单号错了？",
        "我的追踪链接打不开，能换一个吗？",
        "物流系统显示包裹已签收但我没收到",
        "追踪号输入后提示未找到相关记录",
        "为什么我的包裹追踪信息一直不更新？",
        "我用单号查询但系统说查无此单",
        "物流跟踪的地图定位不准，位置不对",
        "追踪信息显示包裹在两个地方同时出现",
        "我输入正确的单号但查不到任何信息",
        "包裹的状态显示已签收但收件人说没收到",
        "查询系统一直加载不出来，是系统故障吗",
        "物流追踪页面是空白的",
        "我的快递单号换了，新单号是多少？",
        "追踪信息说正在派送但没有快递员联系我",
        "包裹在系统中消失了，最后更新是三天前",
        "物流查询APP显示的信息跟网站不一样",
        "我的货柜号查询后显示未知状态",
        "追踪链接过期了，能重新发一个吗",
        "物流系统里我的包裹状态一直卡在运输中",
        "我收到签收通知但实际没人联系我",
        "追踪号查询显示包裹已退回发件地",
        "物流跟踪的预计时间一直不变",
        "我的包裹被错误地送到了另一个地址",
        "追踪系统显示包裹到达了但我没收到通知",
        "输入提单号后查询结果为空",
        "物流信息更新延迟，实际位置跟系统显示不符",
        "我的集装箱追踪系统无法访问",
        "快递追踪显示异常但不知道具体是什么问题",
        "包裹的追踪记录有断层，中间几天没有信息",
        "我的货柜GPS追踪信号中断了",
        "查询系统说包裹已送达但我没有收到",
        "物流跟踪页面显示网络错误",
        "我的追踪号被识别为无效号码",
        "包裹在系统中显示被送到了错误的城市",
        "物流APP上的追踪信息三天没更新了",
        "我的货物在系统中显示已到港但没有提货通知",
        "追踪信息说包裹在仓库但没人知道是哪个仓库",
        "物流查询的结果跟我寄的目的地不一致",
        "我的包裹追踪号重复了，跟别人的单号一样",
        "系统显示货物已清关但我没收到任何通知",
        "我的货柜追踪信息跟船公司给的不一样",
        "物流跟踪的最后一公里信息缺失",
        "查询系统维护中，什么时候恢复？",
        "我的包裹在追踪系统中被标记为丢失",
        "物流信息显示包裹已发出但追踪号查不到",
        "追踪系统说我的货还在起点但实际已经到中途",
        "我的集装箱在追踪地图上消失了",
        "物流跟踪的ETA跟系统显示的不一致",
        "输入运单号后只显示了部分物流信息",
        "我的包裹追踪记录显示被他人代签了",
        "物流系统里看不到国际段的运输信息",
        "追踪信息说等待入库已经一周了",
        "我的快递查询结果是信息待更新",
        "包裹的物流追踪在某个城市后就没有记录了",
        "我的货物追踪号码被系统注销了",
        "物流跟踪页面报错500",
        "我的集装箱号在船公司网站查不到",
        "追踪系统显示已到达但最终目的地不对",
        "物流信息里包裹的重量跟实际不符",
        "我的包裹追踪状态从已签收变成了运输中",
        "查询系统说单号格式不对",
        "物流跟踪没有显示包裹经过了哪些中转站",
        "我的货柜追踪信息停留在出发港",
        "追踪系统无法显示冷链运输的温度记录",
        "物流查询结果显示包裹被拆包了",
    ],
    "billing_dispute": [
        "这个月的运费账单不对，比预期高了很多",
        "我被多收了一笔燃油附加费，能核实吗？",
        "账单上的重量跟我发货时的实际重量不一样",
        "为什么我的运费突然涨价了？",
        "发票金额和合同金额不一致",
        "我被收取了两次的清关费用",
        "运费账单里的体积计算方式有疑问",
        "合同中约定的折扣没有在账单中体现",
        "我的账户被扣了一笔不明费用",
        "运费计算标准是什么？为什么跟报价单不同？",
        "账单显示超重费但我确认没有超重",
        "我被收取了偏远地区附加费但目的地不在偏远区域",
        "月度账单中有一笔重复收费",
        "运费预估和实际收费差距太大",
        "我的VIP折扣没有体现在账单上",
        "为什么旺季附加费比公布的高？",
        "账单上的币种换算有问题",
        "我被收取了仓储费但货物没有超期",
        "合同说免首重但账单上还是收了首重费",
        "运费退款申请提交两周了还没处理",
        "账单上的货物数量跟我实际发的不符",
        "我付了加急费但并没有加急处理",
        "为什么我的运费比同行贵这么多？",
        "账单里多了一笔我不知名的手续费",
        "我的运费月结额度被莫名其妙扣减了",
        "燃油附加费的计算基准是什么？",
        "账单中的目的港杂费是怎么算的？",
        "我被收取了退件费用但我没有退件",
        "运费险的理赔金额跟实际损失不符",
        "合同中约定的运费单价被上调了",
        "我的付款已经扣了但账单还显示欠款",
        "账单上的计费重量是体积重但货物不轻",
        "为什么会有临时附加费？之前没通知",
        "我的账户余额被扣错了，能核对一下吗",
        "账单里的汇率用的是哪天的？",
        "我被收取了目的港THC但合同说已包含",
        "运费计算中为什么有拼箱费？我订的是整箱",
        "月度对账单中有一笔找不到对应的订单",
        "我的运费折扣等级被下调了，为什么没通知？",
        "账单显示我欠费但我记得已经付过了",
        "为什么同一航线的运费我的比别人高？",
        "账单中的报关服务费收了两次",
        "我的预付款被用在了错误的订单上",
        "运费账单中没有列出费用的明细",
        "我被收取了超期保管费但货已经按时提了",
        "汇率波动导致的额外费用谁来承担？",
        "账单上多了一笔改单费但我没有改过单",
        "我的运费年返利到现在还没有到账",
        "为什么我的账户突然产生了滞纳金？",
        "账单中的目的地代码跟实际不一致",
        "我付了人民币但账单按美元结算了",
        "运费计算中缺少了折扣项",
        "我的账单金额比系统自动报价高了30%",
        "为什么我的订单被收取了紧急附加费？",
        "账单中有一笔取消费但我没有取消订单",
        "我的运费补贴申请被拒绝了，能重新审核吗",
        "为什么我的月结账单中有现金支付记录？",
        "账单中的附加费项目没有说明收费原因",
        "我的运费跟三个月前比涨了25%，正常吗？",
        "我被收取了单证费但没有收到任何单证",
        "账单里的费用项目能逐项解释一下吗",
        "我的账户被冻结了但我不欠费",
        "为什么我的退款被扣了手续费？",
        "运费账单中的货物分类被错误地标记了",
        "我的合同价在系统中没有被正确录入",
    ],
    "schedule_inquiry": [
        "下一班到鹿特丹的船期是什么时候？",
        "从上海到洛杉矶的船期表能发我一份吗",
        "五月从宁波出发到汉堡的船有哪些？",
        "这周有到东南亚的航班吗？",
        "从深圳到纽约的货运航班一周几班？",
        "到釜山港的下一班船哪天发？",
        "我想知道从天津到长滩的船期",
        "下周到新加坡的集装箱船有几班？",
        "从青岛到安特卫普的航线船期安排",
        "五月中旬到南美有船吗？",
        "从厦门到迪拜的船期是什么时候",
        "广州到悉尼的海运船期表",
        "到日本的船多久一班？",
        "从大连到温哥华的船期信息",
        "到印度那瓦舍瓦港的船期",
        "从上海到非洲的航线船期",
        "五月从南沙到中东有船吗",
        "到英国的货运航班频率是多少",
        "从连云港到东南亚的船期",
        "下周到韩国的船有几班",
        "从苏州到美国西岸的运输时间",
        "六月从深圳到澳洲的船期",
        "到巴西桑托斯港的船期表",
        "从上海出发经过巴拿马运河的航线",
        "到土耳其伊斯坦布尔的船期",
        "从天津到地中海沿岸的船期",
        "下周有到越南的快船吗",
        "从宁波到东海岸的船期安排",
        "到智利瓦尔帕莱索的航线",
        "从上海到红海的船期信息",
        "七月从青岛到澳大利亚的船",
        "到南非德班的船多久一班",
        "从深圳到加拿大东岸的船期",
        "到菲律宾马尼拉的船期",
        "从上海到挪威的航线",
        "五月到马来西亚的船有几班",
        "从广州到印尼雅加达的船期",
        "到波兰格但斯克的船期",
        "从宁波到泰国林查班的船",
        "到哥伦比亚布埃纳文图拉的船期",
        "从深圳到西班牙瓦伦西亚的航线",
        "下周到柬埔寨的船",
        "从上海到斯里兰卡科伦坡的船期",
        "到阿根廷布宜诺斯艾利斯的航线",
        "从天津到新西兰奥克兰的船期",
        "五月到孟加拉吉大港的船",
        "从青岛到秘鲁卡亚俄的船期",
        "到埃及亚历山大港的航线",
        "从上海到乌拉圭蒙得维的亚的船期",
        "下周到缅甸仰光的船",
        "从深圳到芬兰赫尔辛基的船期",
        "到肯尼亚蒙巴萨的船期",
        "从宁波到坦桑尼亚达累斯萨拉姆的船",
        "五月到克罗地亚里耶卡的船期",
        "从上海到委内瑞拉的航线",
        "到突尼斯的船多久一班",
        "从广州到海防港的船期",
        "到爱尔兰都柏林的船",
        "从天津到斯洛文尼亚科佩尔的船期",
        "五月到智利的船有几班",
        "从上海到黑山的船期",
        "到科特迪瓦阿比让的航线",
        "从深圳到拉脱维亚里加的船期",
        "下周到莫桑比克马普托的船",
        "从青岛到危地马拉的船期",
    ],
    "cargo_damage": [
        "收到的货物外包装破损，里面东西也坏了",
        "集装箱到达时发现门是开的，货物被盗",
        "货物受潮了，包装箱都湿透了",
        "卸货时发现货物有挤压变形",
        "我的货物在运输过程中被摔坏了",
        "收到货发现少了一半，明显被拆过",
        "玻璃制品全部碎裂，包装保护不够",
        "货物被水浸过，外包装有水渍",
        "集装箱内有异味，货物被污染了",
        "到货发现货物数量少了10箱",
        "电子产品到货后无法开机，运输中损坏",
        "货物外包装有明显撞击痕迹",
        "收到的货物跟发出时完全不一样",
        "冷链运输温度不达标，食品已经变质",
        "到货后发现货物被雨淋过",
        "货物在码头上被叉车撞坏了",
        "我的货柜到了但里面的货散落一地",
        "服装类货物到货后发霉了",
        "精密仪器到货后发现精度偏差",
        "货物外包装的封条被破坏过",
        "到货后发现部分货物标签被换了",
        "木材类货物到达后开裂变形",
        "化学品包装破损，有泄漏",
        "货物在运输中被太阳暴晒后变质",
        "到货后发现货物有虫蛀痕迹",
        "金属制品到货后生锈了",
        "纸箱包装的货物全部被压扁了",
        "货物到达后发现内件被调换",
        "液体货物泄漏污染了其他货物",
        "到货后发现货物缺少配件",
        "货物在转运过程中被雨淋了",
        "易碎品到货后全部损坏",
        "货物包装上的防水标签失效了",
        "我的货柜到了但铅封号码不对",
        "货物被堆放在露天场地淋雨了",
        "到货后发现货物有过期迹象",
        "纺织品到货后发现色差严重",
        "货物在船上被海水溅到了",
        "到货后发现部分货物被虫咬",
        "我的货物被错误地冷冻了",
        "货物外包装完好但内部物品损坏",
        "到货后发现货物少了说明书",
        "家具到货后发现表面有划痕",
        "货物在码头暴晒后包装褪色",
        "我的货物被放在了潮湿的仓库里",
        "到货后发现货物有烧焦痕迹",
        "货物被错误的装卸方式损坏了",
        "我的货柜内部有积水",
        "货物到达后发现有老鼠咬痕",
        "我的货物被其他货压坏了",
        "到货后发现货物标签全部脱落",
        "食品类货物到货后发现包装被老鼠咬过",
        "我的货在运输过程中遭遇车祸",
        "到货后发现货物有化学腐蚀痕迹",
        "货物被堆码过高导致下层被压坏",
        "我的货物在海关查验后损坏了",
        "到货后发现货物缺少合格证",
        "货物在搬运过程中被摔了",
        "我的货柜门在运输中开了",
        "到货后发现货物有受潮发霉",
        "货物被错误地跟危险品放在一起",
        "我的货物到达后温度记录异常",
        "到货后发现货物有油污痕迹",
        "货物在海运过程中被盐雾腐蚀",
        "我的货物在拆箱时发现有损坏",
    ],
    "booking_request": [
        "想订一个40HQ从宁波到洛杉矶的舱",
        "需要订下周从上海到汉堡的集装箱",
        "帮我预订一个20尺柜到釜山",
        "想要六月份从深圳到悉尼的船位",
        "订舱：三个40HQ从青岛到鹿特丹",
        "我需要订一个散货拼箱到新加坡",
        "想预订从天津到长滩的整箱舱位",
        "五月底从广州到迪拜的舱位还有吗",
        "需要订冷冻柜，从上海到迈阿密",
        "想订一个开顶柜从宁波到安特卫普",
        "帮我查一下从厦门到汉堡的订舱价格",
        "需要预订下周一到东南亚的散货舱位",
        "订一个平板柜从青岛到休斯顿",
        "我想订40尺高柜从深圳到洛杉矶",
        "需要订从上海到雅加达的拼箱",
        "预订下周从宁波到东京的船位",
        "想订一个20尺普柜到孟买",
        "帮我订从广州到瓦尔帕莱索的整箱",
        "需要订冷藏柜从上海到智利",
        "预订从深圳到伊斯坦布尔的舱位",
        "想订两个40HQ从宁波到费利克斯托",
        "需要订下个月从青岛到悉尼的舱",
        "帮我预订从上海到圣保罗的整柜",
        "想订一个框架柜从天津到鹿特丹",
        "订舱：从深圳到海防的散货",
        "需要订从宁波到那瓦舍瓦的20尺柜",
        "我想订一个40尺柜到哥本哈根",
        "预订从上海到曼谷的拼箱舱位",
        "需要订冷冻柜从广州到墨尔本",
        "帮我订从青岛到蒙巴萨的舱位",
        "想订从深圳到利雅得的整箱",
        "需要订一个开顶柜从宁波到休斯顿",
        "预订从上海到布宜诺斯艾利斯的船位",
        "想订两个20尺柜从天津到汉堡",
        "需要订从广州到拉各斯的舱位",
        "帮我预订从深圳到热那亚的拼箱",
        "想订一个40HQ从上海到温哥华",
        "需要订从青岛到德班的冷藏柜",
        "预订从宁波到卡亚俄的整箱舱位",
        "想订从深圳到格但斯克的散货",
        "需要订一个平板柜从广州到安特卫普",
        "帮我订从上海到科伦坡的舱位",
        "想订40尺高柜从青岛到萨凡纳",
        "需要订从深圳到巴生港的拼箱",
        "预订从宁波到林查班的整箱",
        "想订一个20尺柜从上海到奥克兰",
        "需要订从广州到吉达的舱位",
        "帮我订从青岛到马尼拉的散货",
        "想订冷冻柜从深圳到汉堡",
        "需要订从上海到金斯顿的整箱",
        "预订从宁波到桑托斯的舱位",
        "想订一个40HQ从天津到西雅图",
        "需要订从广州到丹戎不碌的拼箱",
        "帮我订从青岛到查尔斯顿的舱位",
        "想订从深圳到釜山的整箱",
        "需要订一个框架柜从上海到南安普顿",
        "预订从宁波到伊兹密尔的舱位",
        "想订两个20尺柜从广州到长滩",
        "需要订从深圳到杰贝阿里的散货",
        "帮我订从上海到自由镇的冷藏柜",
        "想订40尺柜从青岛到勒阿弗尔",
        "需要订从宁波到阿帕帕的舱位",
        "预订从深圳到皮尔逊的整箱",
        "想订一个开顶柜从广州到汉堡",
        "需要订从上海到考赛斯的拼箱",
    ],
    "general": [
        "你们公司支持哪些支付方式？",
        "怎么联系人工客服？",
        "你们的工作时间是几点到几点？",
        "能提供英文版的报价单吗",
        "我想投诉上次的服务质量",
        "你们的保险服务是怎么收费的",
        "怎么注册企业账户",
        "有没有手机APP可以下载",
        "你们的仓库地址在哪里",
        "能开增值税专票吗",
        "你们的货物保险包含哪些范围",
        "怎么查询我的账户余额",
        "我想了解你们的企业合作折扣",
        "你们的客服热线周末有人吗",
        "能帮忙查一下上次订单的发票吗",
        "你们的危险品运输资质有吗",
        "我想变更收件人的联系方式",
        "你们支持门到门服务吗",
        "怎么申请运费月结",
        "你们的退件政策是什么",
        "能提供货物追踪的API接口吗",
        "我想了解一下你们的海关代理服务",
        "你们的货物最大承重限制是多少",
        "怎么取消一个已经下的订单",
        "你们的公司全称和纳税人识别号是什么",
        "我想了解你们的仓储服务价格",
        "你们支持电子签收吗",
        "怎么修改收件地址",
        "你们的理赔流程是怎样的",
        "我想查一下历史订单",
        "你们的货物运输保险怎么买",
        "怎么下载电子发票",
        "你们的包装服务收费多少",
        "我想了解一下你们的冷链物流能力",
        "你们的提单可以电放吗",
        "怎么绑定企业邮箱接收通知",
        "你们的仓库能提供分拣服务吗",
        "我想查询我的信用额度",
        "你们的报关服务需要额外收费吗",
        "怎么授权其他人管理我的账户",
        "你们的货物在仓库能存放多久",
        "我想了解一下你们的跨境电商物流",
        "你们的系统支持批量导入订单吗",
        "怎么设置自动对账",
        "你们的货物追踪能精确到小时吗",
        "我想了解一下你们的绿色物流计划",
        "你们的系统有API文档吗",
        "怎么申请成为你们的代理商",
        "你们的客服能讲粤语吗",
        "我想了解一下你们的供应链金融",
        "你们的系统支持多语言吗",
        "怎么查看我的积分和优惠",
        "你们的货物在运输中能改目的地吗",
        "我想了解一下你们的碳足迹报告",
        "你们的仓库有温控设备吗",
        "怎么设置到货自动通知",
        "你们的提单修改需要多久",
        "我想了解一下你们的行业解决方案",
        "你们的系统能跟ERP对接吗",
        "怎么申请数据导出",
        "你们的货物保险理赔需要多久",
        "我想了解一下你们的国际快递服务",
        "你们的仓库能提供打托服务吗",
        "怎么查询我的物流月报",
        "你们支持货到付款吗",
    ],
}


def build_dataset():
    """从种子数据构建数据集"""
    texts = []
    labels = []
    for label_idx, (label_name, examples) in enumerate(DATA_SEEDS.items()):
        for text in examples:
            texts.append(text)
            labels.append(label_idx)

    # Shuffle
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


texts, labels = build_dataset()

# 统计
from collections import Counter
label_counts = Counter(labels)
print(f"\n  数据集总计: {len(texts)} 条")
print(f"  类别分布:")
for idx, name in enumerate(LABELS):
    print(f"    {idx} - {name}: {label_counts[idx]} 条")

# 划分 train/test (80/20)
random.seed(42)
indices = list(range(len(texts)))
random.shuffle(indices)
split_point = int(len(texts) * 0.8)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

print(f"  训练集: {len(train_idx)} 条")
print(f"  测试集: {len(test_idx)} 条")

# ============================================================
# Part 2: Tokenize
# ============================================================

print("\n" + "=" * 60)
print("Part 2: Tokenize")
print("=" * 60)

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 64

print(f"\n  模型: {MODEL_NAME}")
print(f"  最大长度: {MAX_LENGTH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize train
print(f"\n  Tokenize 训练集...")
train_texts_sub = [texts[i] for i in train_idx]
train_labels_sub = [labels[i] for i in train_idx]
train_encodings = tokenizer(train_texts_sub, padding="max_length", truncation=True, max_length=MAX_LENGTH)

# Tokenize test
print(f"  Tokenize 测试集...")
test_texts_sub = [texts[i] for i in test_idx]
test_labels_sub = [labels[i] for i in test_idx]
test_encodings = tokenizer(test_texts_sub, padding="max_length", truncation=True, max_length=MAX_LENGTH)

# 转换为 TensorDataset
train_input_ids = torch.tensor(train_encodings["input_ids"])
train_attention = torch.tensor(train_encodings["attention_mask"])
train_labels_tensor = torch.tensor(train_labels_sub)

test_input_ids = torch.tensor(test_encodings["input_ids"])
test_attention = torch.tensor(test_encodings["attention_mask"])
test_labels_tensor = torch.tensor(test_labels_sub)

train_dataset = TensorDataset(train_input_ids, train_attention, train_labels_tensor)
test_dataset = TensorDataset(test_input_ids, test_attention, test_labels_tensor)

# 展示一条样本
print(f"\n  样本展示:")
sample_tokens = tokenizer.convert_ids_to_tokens(train_input_ids[0].tolist())
print(f"    原文: '{train_texts_sub[0]}'")
print(f"    Token: {sample_tokens[:20]}...")
print(f"    input_ids shape: {train_input_ids[0].shape}")

# ============================================================
# Part 3: 构建分类模型
# ============================================================

print("\n" + "=" * 60)
print("Part 3: 构建 BERT 分类模型")
print("=" * 60)


class TextClassifier(nn.Module):
    """基于 BERT 的文本分类器

    结构: BERT → [CLS] → Dropout → Linear → Logits
    """

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls = self.dropout(cls)
        return self.classifier(cls)


num_labels = len(LABELS)
model = TextClassifier(MODEL_NAME, num_labels)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  模型结构:")
print(f"    BERT (base): {model.bert.config.hidden_size}d hidden, 12 layers")
print(f"    Classifier:  Linear({model.bert.config.hidden_size} → {num_labels})")
print(f"  总参数数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

# ============================================================
# Part 4: 训练循环
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 训练循环")
print("=" * 60)

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

print(f"\n  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Loss: CrossEntropyLoss (多分类)")
print(f"  Optimizer: AdamW")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"\n  开始训练...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_input_ids, batch_attention, batch_labels in train_loader:
        optimizer.zero_grad()
        logits = model(batch_input_ids, batch_attention)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_labels).sum().item()
        total += len(batch_labels)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_input_ids, batch_attention, batch_labels in test_loader:
            logits = model(batch_input_ids, batch_attention)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == batch_labels).sum().item()
            val_total += len(batch_labels)
    val_acc = val_correct / val_total

    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ============================================================
# Part 5: 评估
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 模型评估")
print("=" * 60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_input_ids, batch_attention, batch_labels in test_loader:
        logits = model(batch_input_ids, batch_attention)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\n  测试集准确率 (Accuracy): {acc:.4f}")
print(f"  测试集 F1 (weighted):    {f1:.4f}")

# Classification Report
print(f"\n  Classification Report:")
print(f"  {'':>20} | {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print(f"  {'-'*20}-+-" + "-"*45)
report = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True)
for label_name in LABELS:
    row = report[label_name]
    print(f"  {label_name:>20} | {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1-score']:>10.4f} {row['support']:>10.0f}")
print(f"  {'-'*20}-+-" + "-"*45)
macro = report["macro avg"]
weighted = report["weighted avg"]
print(f"  {'macro avg':>20} | {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1-score']:>10.4f} {macro['support']:>10.0f}")
print(f"  {'weighted avg':>20} | {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} {weighted['f1-score']:>10.4f} {weighted['support']:>10.0f}")

# 混淆矩阵
print(f"\n  混淆矩阵 (行=真实, 列=预测):")
cm = confusion_matrix(all_labels, all_preds)
label_short = {
    "shipping_delay": "delay",
    "customs_clearance": "customs",
    "tracking_issue": "tracking",
    "billing_dispute": "billing",
    "schedule_inquiry": "schedule",
    "cargo_damage": "damage",
    "booking_request": "booking",
    "general": "general",
}
short_names = [label_short[n] for n in LABELS]

header = "  " + f"{'真实\\预测':>12} |"
for sn in short_names:
    header += f" {sn:>10}"
print(header)
print("  " + "-" * 14 + "+" + "-" * (11 * len(short_names)))

for i, sn in enumerate(short_names):
    row_str = f"  {sn:>12} |"
    for j in range(len(short_names)):
        row_str += f" {cm[i][j]:>10}"
    print(row_str)

# 推理演示
print(f"\n  推理演示:")
demo_texts = [
    "我的包裹已经晚了一周了，到底什么情况？",
    "下一班到鹿特丹的船期是什么时候？",
    "货物卡在海关了，怎么加快清关速度？",
    "你们公司支持哪些支付方式？",
    "收到的货物外包装破损，里面东西也坏了",
]

for text in demo_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1).item()
    confidence = probs[0][pred_idx].item()
    print(f"  '{text}'")
    print(f"    → {LABELS[pred_idx]} (confidence: {confidence:.4f})")

# 保存
print(f"\n  保存模型和评估结果...")
torch.save(model.state_dict(), "code/phase7/models/logistics_bert_classifier.pt")

label_map = {str(i): name for i, name in enumerate(LABELS)}
with open("code/phase7/models/label_map.json", "w") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

eval_results = {
    "accuracy": round(acc, 4),
    "f1_weighted": round(f1, 4),
    "f1_macro": round(report["macro avg"]["f1-score"], 4),
    "per_class": {name: {
        "precision": round(report[name]["precision"], 4),
        "recall": round(report[name]["recall"], 4),
        "f1-score": round(report[name]["f1-score"], 4),
        "support": int(report[name]["support"]),
    } for name in LABELS},
    "confusion_matrix": cm.tolist(),
    "model_name": MODEL_NAME,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "dataset_size": len(texts),
    "test_size": len(test_idx),
}

with open("code/phase7/models/eval_results.json", "w") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

print(f"  模型已保存到: code/phase7/models/logistics_bert_classifier.pt")
print(f"  评估结果已保存到: code/phase7/models/eval_results.json")
print(f"\n  Step 3 完成!")
